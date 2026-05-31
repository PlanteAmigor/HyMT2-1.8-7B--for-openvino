#!/usr/bin/env python3
"""
Hy-MT2 → OpenVINO IR Converter

Usage:
  # Default (FP32, auto-detect 1.8B)
  python Hy-MT2/convert_to_ov.py

  # Custom path with quantization
  python Hy-MT2/convert_to_ov.py --model Hy-MT2/7B --output Hy-MT2/7B-ov --weight-format int8

  # Quick INT4
  python Hy-MT2/convert_to_ov.py --weight-format int4
"""

import os, sys, time, argparse
import torch, openvino as ov, nncf
from transformers import AutoModelForCausalLM

# ── 语言 ──
_LANG = "zh" if any(x in os.environ.get("LANG", "") for x in ("zh_CN", "zh-", "zh_")) else "en"
if "--lang" in sys.argv:
    i = sys.argv.index("--lang")
    if i + 1 < len(sys.argv): _LANG = sys.argv[i + 1]

def TR(zh, en): return zh if _LANG == "zh" else en


class HunYuanOVWrapper(torch.nn.Module):
    """
    Wrapper 绕过 HunYuan 模型的 torch.vmap 因果掩码（无法被 jit.trace 追踪）。
    手动构造 4D causal mask 替代。
    """
    def __init__(self, model):
        super().__init__()
        self.layers = model.model.layers
        self.embed_tokens = model.model.embed_tokens
        self.rotary_emb = model.model.rotary_emb
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        device = input_ids.device
        dtype = inputs_embeds.dtype

        causal_mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            causal_mask = causal_mask.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        cache_position = torch.arange(seq_len, device=device)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, attention_mask=causal_mask,
                position_ids=position_ids, past_key_values=None,
                use_cache=False, cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def convert_model(model_path, output_path, weight_format):
    print(f"{TR('模型', 'Model')}: {model_path}")
    print(f"{TR('输出', 'Output')}: {output_path}")
    print(f"{TR('量化', 'Quant')}: {weight_format}")
    print()

    print(TR("[1/4] 加载 PyTorch 模型...", "[1/4] Loading PyTorch model..."), end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, trust_remote_code=True,
    )
    model.config._attn_implementation = "eager"
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ ({time.time()-t0:.1f}s)  {n_params/1e9:.2f}B {TR('参数', 'params')}")

    print(TR("[2/4] 创建 wrapper...", "[2/4] Creating wrapper..."), end=" ", flush=True)
    wrapper = HunYuanOVWrapper(model)
    wrapper.eval()
    print("✓")

    print(TR("[3/4] OpenVINO IR 转换...", "[3/4] Converting to OpenVINO IR..."), end=" ", flush=True)
    example = {"input_ids": torch.ones((1, 1), dtype=torch.long),
               "attention_mask": torch.ones((1, 1), dtype=torch.long)}
    t0 = time.time()
    ov_model = ov.convert_model(wrapper, example_input=example)
    print(f"✓ ({time.time()-t0:.1f}s)")

    if weight_format in ("int8", "int4"):
        print(f"[4/4] {weight_format.upper()} {TR('量化...', 'quantizing...')}", end=" ", flush=True)
        t0 = time.time()
        mode = nncf.CompressWeightsMode.INT4_ASYM if weight_format == "int4" else nncf.CompressWeightsMode.INT8_ASYM
        kwargs = {"group_size": 128} if weight_format == "int4" else {}
        ov_model = nncf.compress_weights(ov_model, mode=mode, **kwargs)
        print(f"✓ ({time.time()-t0:.1f}s)")
    else:
        print(TR("[4/4] 跳过量化 (FP32)", "[4/4] Skipping quantization (FP32)"))

    # ── 保存 ──
    os.makedirs(output_path, exist_ok=True)
    model_xml = os.path.join(output_path, "openvino_model.xml")
    ov.save_model(ov_model, model_xml)
    bin_size = os.path.getsize(model_xml.replace(".xml", ".bin")) / (1024 * 1024)
    print(f"\n{TR('保存到', 'Saved to')}: {output_path}  ({bin_size:.0f} MB)")
    print(TR("完成!", "Done!"))


def main():
    parser = argparse.ArgumentParser(description="Hy-MT2 → OpenVINO IR 转换")
    parser.add_argument("--model", default=None, help="原始模型路径 (默认 auto-detect)")
    parser.add_argument("--output", "-o", default=None, help="输出路径 (默认 {model}-ov)")
    parser.add_argument("--weight-format", choices=["fp32", "int8", "int4"], default="fp32",
                        help="权重量化格式 (默认 fp32)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model or os.path.join(script_dir, "1.8B")

    # 相对路径补全
    if not os.path.isdir(model_path):
        candidate = os.path.join(script_dir, model_path)
        if os.path.isdir(candidate):
            model_path = candidate

    if not os.path.isdir(model_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')} {model_path}")
        sys.exit(1)

    model_name = os.path.basename(os.path.normpath(model_path))
    output_path = args.output or os.path.join(script_dir, f"{model_name}-ov")
    convert_model(model_path, output_path, args.weight_format)


if __name__ == "__main__":
    main()
