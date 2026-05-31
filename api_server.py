#!/usr/bin/env python3
"""
Hy-MT2 1.8B OpenVINO 翻译 API 服务器
支持：流式输出 (SSE)、NPU/GPU/CPU 自动选择、详细日志
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import numpy as np
import openvino as ov
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# ── 日志配置 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hy-mt2")

# ── 常量 ──────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OV_MODEL_DIR = os.path.join(MODEL_DIR, "1.8B-ov")
ORIG_MODEL_DIR = os.path.join(MODEL_DIR, "1.8B")

# 特殊 token ID
BOS_ID = 120000
EOS_ID = 120020
PAD_ID = 120002
USER_ID = 120006
ASST_ID = 120007

# ── 数据结构（翻译 API）──────────────────────────────────
class TranslateRequest(BaseModel):
    text: str = Field(..., description="待翻译文本")
    source_lang: str = Field(default="", description="源语言（可选，自动检测）")
    target_lang: str = Field(default="中文", description="目标语言")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成长度")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(default=0.6, ge=0.0, le=1.0, description="Top-p 采样")
    top_k: int = Field(default=20, ge=1, le=100, description="Top-k 采样")
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0, description="重复惩罚")
    stream: bool = Field(default=True, description="是否流式输出")
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示（覆盖默认翻译指令）")

class ModelInfo(BaseModel):
    model: str = "Hy-MT2-1.8B"
    device: str = ""
    backend: str = "OpenVINO"
    parameters: str = "1.79B"
    max_tokens: int = 4096

# ── 数据结构（OpenAI 兼容 API）───────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(default="user", description="消息角色: system/user/assistant")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="Hy-MT2-1.8B", description="模型名称")
    messages: list[ChatMessage] = Field(..., description="对话消息列表")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成长度")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="采样温度（0=贪心，翻译推荐）")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p 采样")
    stream: bool = Field(default=False, description="是否流式输出")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    stop: Optional[list[str]] = Field(default=None, description="停止序列")
    user: Optional[str] = Field(default=None, description="用户标识")

# ── OpenVINO 推理引擎 ──────────────────────────────────────
@dataclass
class OVInferenceEngine:
    """封装 OpenVINO 模型的加载、编译和推理"""
    compiled_model: any = None
    tokenizer: any = None
    device: str = ""
    use_fp16: bool = False

    # 内部状态
    _input_ids_key: str = "input_ids"
    _attention_mask_key: str = ""
    _output_key: any = None

    def load(self, device: str = "AUTO"):
        """加载模型和 tokenizer"""
        # 加载 tokenizer
        log.info(f"加载 tokenizer 从 {ORIG_MODEL_DIR} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            ORIG_MODEL_DIR, trust_remote_code=True, fix_mistral_regex=True
        )
        log.info(f"✓ Tokenizer 加载完成 ({time.time()-t0:.1f}s)")
        log.info(f"  Vocab 大小: {self.tokenizer.vocab_size}")
        log.info(f"  特殊 token: BOS={BOS_ID}, EOS={EOS_ID}, PAD={PAD_ID}")

        # 加载 OpenVINO 模型
        log.info(f"加载 OpenVINO 模型从 {OV_MODEL_DIR} ...")
        core = ov.Core()

        ov_model_path = os.path.join(OV_MODEL_DIR, "openvino_model.xml")
        if not os.path.exists(ov_model_path):
            log.error(f"模型文件不存在: {ov_model_path}")
            log.error("请先运行转换脚本: python Hy-MT2/convert_to_ov.py")
            sys.exit(1)

        t0 = time.time()
        ov_model = core.read_model(ov_model_path)
        log.info(f"✓ 模型读取完成 ({time.time()-t0:.1f}s)")

        # 检查设备支持并选择最佳设备
        available = core.available_devices
        log.info(f"可用设备: {available}")

        if device == "AUTO":
            # GPU 优先（用户指定）
            preferred = ["GPU", "CPU"]
            for dev in preferred:
                if dev in available:
                    device = dev
                    break
            else:
                device = "CPU"

        self.device = device
        log.info(f"目标推理设备: {device}")

        # 按优先级尝试编译：目标设备 → CPU
        self._try_compile(core, ov_model, device)

        # 确定输入输出 key
        for i, inp in enumerate(self.compiled_model.inputs):
            name = inp.get_any_name()
            shape = inp.partial_shape
            log.info(f"  输入[{i}]: {name} shape={shape}")

        for i, out in enumerate(self.compiled_model.outputs):
            shape = out.partial_shape
            log.info(f"  输出[{i}]: shape={shape}")

        self._output_key = list(self.compiled_model.outputs)[0]
        log.info(f"✓ 推理引擎就绪 (设备={self.device})")

    def _try_compile(self, core, ov_model, preferred_device):
        """尝试按优先级编译模型"""
        # 确定尝试顺序
        devices_to_try = [preferred_device]
        for fallback in ["NPU", "GPU", "CPU"]:
            if fallback != preferred_device:
                devices_to_try.append(fallback)

        last_error = None
        for dev in devices_to_try:
            if dev not in core.available_devices:
                continue

            log.info(f"尝试编译到 {dev} ...")
            t0 = time.time()
            try:
                if dev == "GPU":
                    self.compiled_model = core.compile_model(ov_model, "GPU")
                else:  # CPU
                    self.compiled_model = core.compile_model(ov_model, "CPU")

                log.info(f"  ✓ {dev} 编译成功! ({time.time()-t0:.1f}s)")
                self.device = dev
                return
            except Exception as e:
                elapsed = time.time() - t0
                log.warning(f"  ✗ {dev} 编译失败 ({elapsed:.1f}s): {str(e)[:120]}")
                last_error = e

        # 全部失败
        log.error("所有设备编译均失败!")
        raise RuntimeError(f"无法编译模型: {last_error}")

    def build_translation_prompt(self, text: str, target_lang: str,
                                  source_lang: str = "", system_prompt: str = None) -> str:
        """构造翻译提示"""
        if system_prompt:
            prompt = system_prompt
        else:
            # 默认翻译指令
            if any('\u4e00' <= c <= '\u9fff' for c in text[:20]):
                # 中文源文本 → 用中文指令
                prompt = f"将以下文本翻译为{target_lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
            else:
                # 非中文 → 用英文指令
                prompt = f"Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:\n\n{text}"

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def tokenize(self, text: str) -> tuple:
        """将文本转为模型输入"""
        inputs = self.tokenizer(text, return_tensors="np", padding=True)
        return inputs["input_ids"].astype(np.int64), inputs["attention_mask"].astype(np.int64)

    def decode(self, token_ids) -> str:
        """将 token ID 解码为文本"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_single(self, input_ids: np.ndarray, attention_mask: np.ndarray,
                        max_new_tokens: int = 512) -> list:
        """单次推理（非流式），返回所有生成的 token ID 列表"""
        generated = []
        self._last_ids = []

        for step in range(max_new_tokens):
            # 推理
            outputs = self.compiled_model([input_ids, attention_mask])
            logits = outputs[self._output_key]

            # 取最后一个位置的 logits
            next_logits = logits[0, -1, :]

            # 温度==0 用贪心，否则用采样
            if self._temperature <= 0:
                next_id = self._greedy(next_logits)
            else:
                next_id = self._sample(next_logits)

            generated.append(int(next_id))
            self._last_ids.append(int(next_id))

            # 检查结束
            if next_id == EOS_ID:
                break

            # 拼接全部历史（OpenVINO 无内部 KV-cache）
            input_ids = np.concatenate([
                input_ids, np.array([[int(next_id)]], dtype=np.int64)
            ], axis=1)
            attention_mask = np.concatenate([
                attention_mask, np.array([[1]], dtype=np.int64)
            ], axis=1)

        return generated

    def generate_stream(self, input_ids: np.ndarray, attention_mask: np.ndarray,
                        max_new_tokens: int = 512):
        """流式生成，逐个 token 产出"""
        generated_text = ""
        step_times = []

        for step in range(max_new_tokens):
            t0 = time.time()

            # 推理
            outputs = self.compiled_model([input_ids, attention_mask])
            logits = outputs[self._output_key]

            # 取最后一个位置
            next_logits = logits[0, -1, :]

            # 温度==0 用贪心，否则用采样
            if self._temperature <= 0:
                next_id = self._greedy(next_logits)
            else:
                next_id = self._sample(next_logits)

            t1 = time.time()
            step_times.append(t1 - t0)

            # 解码当前 token
            token_text = self.tokenizer.decode([next_id], skip_special_tokens=True)
            generated_text += token_text

            # 记录生成的 ID 用于重复惩罚
            if not hasattr(self, '_last_ids'):
                self._last_ids = []
            self._last_ids.append(int(next_id))

            yield {
                "token_id": int(next_id),
                "token": token_text,
                "full_text": generated_text,
                "step": step + 1,
                "step_time_ms": round((t1 - t0) * 1000, 1),
                "finished": bool(next_id == EOS_ID),
            }

            # 检查结束
            if next_id == EOS_ID:
                break

            # 拼接全部历史 token（OpenVINO 模型无内部 KV-cache）
            input_ids = np.concatenate([
                input_ids, np.array([[int(next_id)]], dtype=np.int64)
            ], axis=1)
            attention_mask = np.concatenate([
                attention_mask, np.array([[1]], dtype=np.int64)
            ], axis=1)

        # 统计
        if step_times:
            avg_ms = sum(step_times) / len(step_times) * 1000
            total_s = sum(step_times)
            tokens = len(step_times)
            log.info(f"  生成 {tokens} tokens | 平均 {avg_ms:.1f}ms/token | "
                     f"总计 {total_s:.1f}s | {tokens/total_s:.1f} tok/s")

    def _greedy(self, logits: np.ndarray) -> int:
        """贪心解码"""
        return int(np.argmax(logits).item())

    def _sample(self, logits: np.ndarray) -> int:
        """带重复惩罚的采样"""
        logits = logits.copy().astype(np.float64)

        # 重复惩罚
        if self._repetition_penalty > 1.0 and hasattr(self, '_last_ids') and self._last_ids:
            for prev_id in set(self._last_ids[-100:]):
                if logits[prev_id] > 0:
                    logits[prev_id] /= self._repetition_penalty
                else:
                    logits[prev_id] *= self._repetition_penalty

        # 温度
        if self._temperature > 0 and self._temperature != 1.0:
            logits = logits / self._temperature

        # Top-K
        if self._top_k > 0:
            top_k = min(self._top_k, len(logits))
            top_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.ones(len(logits), dtype=bool)
            mask[top_indices] = False
            logits[mask] = float("-inf")

        # Top-P
        if self._top_p < 1.0:
            sorted_idx = np.argsort(logits)[::-1]
            cumsum = np.cumsum(np.exp(logits[sorted_idx]) / np.sum(np.exp(logits[sorted_idx])))
            cutoff = int(np.searchsorted(cumsum, self._top_p)) + 1
            mask = np.ones(len(logits), dtype=bool)
            mask[sorted_idx[:cutoff]] = False
            logits[mask] = float("-inf")

        # Softmax
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return int(np.random.choice(len(probs), p=probs))

    def set_sampling_params(self, temperature=0.7, top_p=0.6, top_k=20,
                            repetition_penalty=1.05):
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._repetition_penalty = repetition_penalty
        self._last_ids = []


# ── 全局引擎实例 ──────────────────────────────────────────
engine: OVInferenceEngine = None


# ── FastAPI 应用 ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    log.info("=" * 60)
    log.info("Hy-MT2 翻译 API 服务器启动")
    log.info("=" * 60)

    # 加载引擎（自动选择: GPU 优先，无 GPU 则 CPU）
    engine = OVInferenceEngine()
    engine.load(device="AUTO")

    log.info("=" * 60)
    log.info("服务器就绪，等待请求...")
    log.info(f"API 地址: http://{host}:{port}")
    log.info(f"API 文档: http://{host}:{port}/docs")
    log.info("=" * 60)
    yield

    log.info("服务器关闭")


app = FastAPI(
    title="Hy-MT2 翻译 API",
    description="基于 OpenVINO 的 Hy-MT2-1.8B 翻译服务，支持流式输出",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "service": "Hy-MT2 Translation API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/v1/models")
async def get_models():
    """获取模型信息（OpenAI 兼容格式）"""
    global engine
    models = [
        {"id": "Hy-MT2-1.8B", "object": "model", "created": 1748678400,
         "owned_by": "Tencent-Hunyuan", "permission": [], "root": "Hy-MT2-1.8B"},
        {"id": "HyMT2", "object": "model", "created": 1748678400,
         "owned_by": "Tencent-Hunyuan", "permission": [], "root": "HyMT2"},
        {"id": "gpt-3.5-turbo", "object": "model", "created": 1748678400,
         "owned_by": "Tencent-Hunyuan", "permission": [], "root": "gpt-3.5-turbo"},
    ]
    return {"object": "list", "data": models}


@app.post("/v1/translate")
async def translate(request: TranslateRequest):
    """翻译 API - 非流式"""
    global engine
    if not engine:
        raise HTTPException(503, "模型未加载")

    log.info(f"[请求] 翻译文本: {request.text[:80]}... → {request.target_lang}")
    log.info(f"  参数: temp={request.temperature}, top_p={request.top_p}, "
             f"top_k={request.top_k}, rep_penalty={request.repetition_penalty}")

    # 构造提示
    prompt = engine.build_translation_prompt(
        request.text, request.target_lang, request.source_lang, request.system_prompt
    )
    log.info(f"  提示: {prompt[:120]}...")

    # Tokenize
    input_ids, attn_mask = engine.tokenize(prompt)
    prompt_len = input_ids.shape[1]
    log.info(f"  输入 tokens: {prompt_len}")

    # 设置采样参数
    engine.set_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
    )

    # 生成
    t0 = time.time()
    if request.stream:
        return StreamingResponse(
            stream_response(request, input_ids, attn_mask),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        tokens = engine.generate_single(input_ids, attn_mask, request.max_tokens)
        elapsed = time.time() - t0

        result = engine.decode(tokens)
        log.info(f"[完成] 生成 {len(tokens)} tokens, 耗时 {elapsed:.1f}s, "
                 f"{len(tokens)/elapsed:.1f} tok/s")
        log.info(f"[结果] {result[:200]}")

        return {
            "translated_text": result,
            "tokens_generated": len(tokens),
            "tokens_input": prompt_len,
            "time_seconds": round(elapsed, 2),
            "tokens_per_second": round(len(tokens) / elapsed, 1) if elapsed > 0 else 0,
        }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容：聊天补全（翻译模型）"""
    global engine
    try:
        if not engine:
            raise HTTPException(503, "模型未加载")

        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        # 校验 messages（空消息返回空结果）
        if not request.messages:
            log.warning(f"⚠️ 收到空 messages 请求，model={request.model}, stream={request.stream}")
            return _empty_chat_response(completion_id, created, request)

        log.info(f"[OpenAI] messages[{len(request.messages)}]: role={request.messages[0].role}, content_preview={request.messages[0].content[:100]}")
        log.info(f"[OpenAI] model={request.model}, stream={request.stream}")

        # 直接使用 chat template
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        log.info(f"[OpenAI] 请求: {request.messages[-1].content[:80]}...")
        log.info(f"[OpenAI] 参数: temp={request.temperature}, top_p={request.top_p}, "
                 f"stream={request.stream}, max_tokens={request.max_tokens}")

        # Tokenize
        input_ids, attn_mask = engine.tokenize(prompt)
        prompt_len = input_ids.shape[1]

        # 采样参数
        rep_penalty = 1.0 + request.frequency_penalty if request.frequency_penalty > 0 else 1.05
        engine.set_sampling_params(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=20,
            repetition_penalty=rep_penalty,
        )

        if request.stream:
            return StreamingResponse(
                _openai_stream(completion_id, created, request, input_ids, attn_mask, prompt_len),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        else:
            t0 = time.time()
            tokens = engine.generate_single(input_ids, attn_mask, request.max_tokens)
            elapsed = time.time() - t0
            result = engine.decode(tokens) if tokens else ""

            log.info(f"[OpenAI] 完成: {len(tokens)} tokens, {elapsed:.1f}s, "
                     f"{len(tokens)/elapsed:.1f} tok/s" if tokens else "[OpenAI] 完成: 0 tokens")
            if result:
                log.info(f"[OpenAI] 结果: {result[:200]}")

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result},
                        "finish_reason": "stop" if (tokens and tokens[-1] == 120020) else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": len(tokens),
                    "total_tokens": prompt_len + len(tokens),
                },
            }
    except Exception as e:
        log.error(f"[OpenAI] 错误: {e}")
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model if request else "Hy-MT2-1.8B",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def _empty_chat_response(completion_id: str, created: int, request: ChatCompletionRequest):
    """返回空聊天响应（用于连接测试）"""
    # 连接测试返回一个简单问候，Pot 可能因空内容判失败
    msg = "I'm ready! Send me a translation task."
    if request.stream:
        async def _empty_stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': msg}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty_stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    return {
        "id": completion_id, "object": "chat.completion", "created": created, "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": msg}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _openai_stream(completion_id: str, created: int, request: ChatCompletionRequest,
                          input_ids, attn_mask, prompt_len) -> AsyncGenerator[str, None]:
    """OpenAI 兼容：流式 SSE 响应"""
    global engine
    t_start = time.time()
    token_count = 0

    # 角色消息
    yield f"data: {json.dumps({
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': request.model,
        'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}],
    })}\n\n"

    for chunk in engine.generate_stream(input_ids, attn_mask, request.max_tokens):
        token_count += 1

        if chunk["finished"]:
            yield f"data: {json.dumps({
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': request.model,
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
            })}\n\n"
            break

        yield f"data: {json.dumps({
            'id': completion_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': request.model,
            'choices': [{'index': 0, 'delta': {'content': chunk['token']}, 'finish_reason': None}],
        }, ensure_ascii=False)}\n\n"

    elapsed = time.time() - t_start
    log.info(f"[OpenAI 流式] 完成: {token_count} tokens, {elapsed:.1f}s, "
             f"{token_count/elapsed:.1f} tok/s")

    yield "data: [DONE]\n\n"


async def stream_response(request: TranslateRequest, input_ids, attn_mask) -> AsyncGenerator[str, None]:
    """流式 SSE 响应"""
    global engine

    engine.set_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
    )

    prompt_len = input_ids.shape[1]
    t_start = time.time()
    token_count = 0

    # 发送开始事件
    yield f"data: {json.dumps({'event': 'start', 'tokens_input': prompt_len})}\n\n"

    for chunk in engine.generate_stream(input_ids, attn_mask, request.max_tokens):
        token_count += 1
        data = {
            "event": "token",
            "token": chunk["token"],
            "full_text": chunk["full_text"],
            "step": chunk["step"],
            "step_time_ms": chunk["step_time_ms"],
            "finished": chunk["finished"],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        if chunk["finished"]:
            break

    elapsed = time.time() - t_start
    log.info(f"[流式完成] 生成 {token_count} tokens, 耗时 {elapsed:.1f}s, "
             f"{token_count/elapsed:.1f} tok/s")

    # 发送完成事件
    yield f"data: {json.dumps({
        'event': 'done',
        'tokens_generated': token_count,
        'time_seconds': round(elapsed, 2),
        'tokens_per_second': round(token_count / elapsed, 1) if elapsed > 0 else 0,
    })}\n\n"
    yield "data: [DONE]\n\n"


# ── 入口 ──────────────────────────────────────────────────
# ── 命令行参数（在 import 时设置默认，供 lifespan 使用） ──
host = "0.0.0.0"
port = 8000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hy-MT2 OpenVINO 翻译 API 服务器")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()

    host = args.host
    port = args.port

    log.info(f"启动 Hy-MT2 翻译 API 服务: http://{host}:{port}")
    log.info(f"API 文档: http://{host}:{port}/docs")
    log.info(f"模型: Hy-MT2-1.8B | OpenVINO 后端")

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
