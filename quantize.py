#!/usr/bin/env python3
"""
Hy-MT2 OpenVINO 模型量化脚本
===========================
用法: python quantize.py [int8|int4|int4_gs128]
"""

import os, sys, time, shutil
import openvino as ov
import nncf
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(MODEL_DIR, "1.8B-ov")
MODEL_XML = os.path.join(SRC_DIR, "openvino_model.xml")

MODES = {
    "int8":       (nncf.CompressWeightsMode.INT8_ASYM, None,     "1.8B-ov-int8"),
    "int4":       (nncf.CompressWeightsMode.INT4_ASYM, None,     "1.8B-ov-int4"),
    "int4_gs128": (nncf.CompressWeightsMode.INT4_ASYM, 128,      "1.8B-ov-int4-gs128"),
}

help_text = f"""
用法: python quantize.py <模式>
模式: {' | '.join(MODES.keys())}
示例: python quantize.py int4_gs128
"""

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(help_text)
        sys.exit(1)

    mode_name = sys.argv[1]
    mode, group_size, out_dir_name = MODES[mode_name]
    out_dir = os.path.join(MODEL_DIR, out_dir_name)

    print(f"{'='*50}")
    print(f"Hy-MT2 量化: {mode_name}")
    print(f"{'='*50}")

    # 读取原始模型
    print(f"\n[1/3] 读取模型 ...")
    t0 = time.time()
    core = ov.Core()
    model = core.read_model(MODEL_XML)
    orig_bin = os.path.join(SRC_DIR, "openvino_model.bin")
    orig_size = os.path.getsize(orig_bin)
    print(f"  原始大小: {orig_size/1024/1024:.0f} MB ({time.time()-t0:.1f}s)")

    # 压缩
    print(f"\n[2/3] 权重压缩: mode={mode_name}", end="")
    if group_size:
        print(f", group_size={group_size}", end="")
    print(" ...")
    t0 = time.time()
    kwargs = {"mode": mode}
    if group_size:
        kwargs["group_size"] = group_size
    compressed = nncf.compress_weights(model, **kwargs)
    t1 = time.time()
    print(f"  压缩完成 ({t1-t0:.1f}s)")

    # 保存
    print(f"\n[3/3] 保存到 {out_dir} ...")
    os.makedirs(out_dir, exist_ok=True)
    out_xml = os.path.join(out_dir, "openvino_model.xml")
    ov.save_model(compressed, out_xml)
    new_bin = out_xml.replace(".xml", ".bin")
    new_size = os.path.getsize(new_bin)
    ratio = new_size / orig_size * 100
    print(f"  量化后大小: {new_size/1024/1024:.0f} MB ({ratio:.0f}%)")

    # 复制 tokenizer 等配置
    for f in os.listdir(SRC_DIR):
        if f.endswith(".json") or f.endswith(".jinja"):
            shutil.copy2(os.path.join(SRC_DIR, f), os.path.join(out_dir, f))

    # GPU 验证
    print(f"\n  GPU 验证 ...")
    try:
        compiled = core.compile_model(compressed, "GPU")
        input_ids = np.array([[120000]], dtype=np.int64)
        mask = np.array([[1]], dtype=np.int64)
        out = compiled([input_ids, mask])
        logits = list(out.values())[0]
        pred = int(np.argmax(logits[0, -1]))
        print(f"  ✅ GPU 推理成功, 预测 token: {pred}")
    except Exception as e:
        print(f"  ⚠️ GPU 推理失败: {e}")

    print(f"\n{'='*50}")
    print(f"完成! 输出目录: {out_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
