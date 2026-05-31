#!/usr/bin/env python3
"""
Hy-MT2 1.8B OpenVINO GPU Terminal Chat
========================================
Auto language detection: Chinese→English, otherwise→Chinese
//en text  → force English,  //zh text  → force Chinese
/help for help, /exit to quit
"""

import os
import sys
import time
import logging
import readline
import numpy as np
import openvino as ov
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hy-mt2")
log.setLevel(logging.WARNING)

# ── 语言检测 ──
_LANG = os.environ.get("LANG", "en_US")
if any(x in _LANG for x in ("zh_CN", "zh-", "zh_")):
    _LANG = "zh"
else:
    _LANG = "en"
# --lang 参数覆盖
if "--lang" in sys.argv:
    i = sys.argv.index("--lang")
    if i + 1 < len(sys.argv):
        _LANG = sys.argv[i + 1]

def TR(zh, en):
    return zh if _LANG == "zh" else en

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 模型路径：环境变量 > --ov-path 参数 > 默认 ──
_ov_path = os.environ.get("HY_MT2_OV_PATH", "")
if not _ov_path and "--ov-path" in sys.argv:
    i = sys.argv.index("--ov-path")
    if i + 1 < len(sys.argv):
        _ov_path = sys.argv[i + 1]
OV_PATH = _ov_path or os.path.join(MODEL_DIR, "1.8B-ov")
ORIG_PATH = _ov_path or os.path.join(MODEL_DIR, "1.8B")
EOS_TOKEN = 120020

# ── 翻译模板 ──
T_ZH = "将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
T_EN = "Translate the following text into {target}. Note that you should only output the translated result without any additional explanation:\n\n{text}"

def has_chinese(text):
    return any('\u4e00' <= c <= '\u9fff' for c in text[:30])

def read_multiline(prompt=">>> "):
    """读取用户输入，支持多行：输入三个双引号进入多行模式，再输入三个双引号结束。"""
    line = input(prompt).strip()
    # 多行模式：以 """ 开始
    if line == '"""':
        lines = []
        while True:
            try:
                l = input()
            except EOFError:
                break
            if l.strip() == '"""':
                break
            lines.append(l)
        return "\n".join(lines)
    # 普通单行模式
    return line

# ── 加载模型 ──
_model_name = os.path.basename(os.path.normpath(OV_PATH))
print(TR(f"加载 Hy-MT2 {_model_name} ...", f"Loading Hy-MT2 {_model_name} ..."), end=" ", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(ORIG_PATH, trust_remote_code=True, fix_mistral_regex=True)
core = ov.Core()
ov_model = core.read_model(os.path.join(OV_PATH, "openvino_model.xml"))
device = "GPU" if "GPU" in core.available_devices else "CPU"
print(f"({device})", end=" ", flush=True)
compiled = core.compile_model(ov_model, device)
print(f"✓ ({time.time()-t0:.1f}s)")

# ── 推理 ──
def translate(text, target_lang):
    prompt = T_ZH.format(target=target_lang, text=text) if has_chinese(text) else T_EN.format(target=target_lang, text=text)
    messages = [{"role": "user", "content": prompt}]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"].astype(np.int64)
    attn_mask = inputs["attention_mask"].astype(np.int64)
    generated = []
    for _ in range(512):
        outputs = compiled([input_ids, attn_mask])
        next_id = int(np.argmax(list(outputs.values())[0][0, -1, :]))
        if next_id == EOS_TOKEN:
            break
        generated.append(next_id)
        input_ids = np.concatenate([input_ids, np.array([[next_id]], dtype=np.int64)], axis=1)
        attn_mask = np.concatenate([attn_mask, np.array([[1]], dtype=np.int64)], axis=1)
    return tokenizer.decode(generated, skip_special_tokens=True)

def translate_stream(text, target_lang):
    """流式生成器，逐个 token 产出翻译结果。"""
    prompt = T_ZH.format(target=target_lang, text=text) if has_chinese(text) else T_EN.format(target=target_lang, text=text)
    messages = [{"role": "user", "content": prompt}]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"].astype(np.int64)
    attn_mask = inputs["attention_mask"].astype(np.int64)
    for _ in range(512):
        outputs = compiled([input_ids, attn_mask])
        next_id = int(np.argmax(list(outputs.values())[0][0, -1, :]))
        if next_id == EOS_TOKEN:
            break
        yield next_id
        input_ids = np.concatenate([input_ids, np.array([[next_id]], dtype=np.int64)], axis=1)
        attn_mask = np.concatenate([attn_mask, np.array([[1]], dtype=np.int64)], axis=1)

# ── 主循环 ──
def main():
    _dev = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
    print()
    print("=" * 50)
    print("  Hy-MT2 " + TR("翻译终端", "Translation Terminal"))
    print(f"  {TR('设备', 'Device')}: {_dev} | OpenVINO")
    print("=" * 50)
    print("  " + TR("直接输入文本 → 自动检测翻译方向", "Type text → auto detect language"))
    print("  //en " + TR("文本 → 强制译英", "text → force English"))
    print("  //zh " + TR("文本 → 强制译中", "text → force Chinese"))
    print("  /help " + TR("→ 帮助", "→ help"))
    print("  /exit " + TR("→ 退出", "→ quit"))
    print("=" * 50)
    print()

    while True:
        try:
            text = read_multiline(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text in ("/exit", "exit", TR("退出", "exit")):
            break
        if text in ("/help", "help", TR("帮助", "help")):
            print("  " + TR("用法：", "Usage:"))
            print("    " + TR("今天天气真好", "Good weather today") + "          → " + TR("自动→英语", "auto→English"))
            print("    Hello world           → " + TR("自动→中文", "auto→Chinese"))
            print("    //en " + TR("你好", "Hello") + "             → " + TR("强制译英", "force English"))
            print("    //zh " + TR("Hello", "Hello") + "          → " + TR("强制译中", "force Chinese"))
            print('    """                   → ' + TR("多行输入", "multi-line input"))
            print("    /exit                 " + TR("退出", "exit"))
            print()
            continue

        force_target = None
        if text.startswith("//en "):
            force_target = TR("英语", "English")
            text = text[5:]
        elif text.startswith("//zh "):
            force_target = TR("中文", "Chinese")
            text = text[5:]
        elif text.startswith("//"):
            print("  ⚠ " + TR("未知指令，可用 //en 或 //zh", "Unknown command, use //en or //zh"))
            continue

        if force_target:
            target = force_target
        elif has_chinese(text):
            target = TR("英语", "English")
        else:
            target = TR("中文", "Chinese")

        print(f"  → {target}  (temp=0)", flush=True)
        t0 = time.time()
        # 流式输出
        sys.stdout.write("  ")
        sys.stdout.flush()
        for token_id in translate_stream(text, target):
            chunk = tokenizer.decode([token_id], skip_special_tokens=True)
            sys.stdout.write(chunk)
            sys.stdout.flush()
        elapsed = time.time() - t0
        print()
        print(f"  [{elapsed:.1f}s]")
        print()

if __name__ == "__main__":
    main()
