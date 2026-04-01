#!/usr/bin/env python3
"""测试 transformers 加载 VibeVoice-Realtime"""

import os
import sys

model_path = os.path.expanduser("~/models/VibeVoice-Realtime-0.5B")

print("=" * 60)
print("VibeVoice-Realtime - Transformers 加载测试")
print("=" * 60)

print(f"\n📂 模型路径：{model_path}")

# 测试 1: Tokenizer
print("\n1️⃣ 测试 Tokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"   ✅ Tokenizer 加载成功")
    print(f"   词表大小：{tokenizer.vocab_size}")
except Exception as e:
    print(f"   ❌ Tokenizer 失败：{e}")

# 测试 2: 模型
print("\n2️⃣ 测试模型加载...")
try:
    from transformers import AutoModel
    print("   使用 AutoModel...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="mps" if sys.platform == "darwin" else "cpu"
    )
    print(f"   ✅ 模型加载成功")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   参数量：{param_count / 1e9:.2f}B")
except Exception as e:
    print(f"   ❌ 模型加载失败：{e}")
    print(f"\n💡 VibeVoice-Realtime 需要官方推理代码")
    print(f"   参考：https://github.com/microsoft/VibeVoice/tree/main")

print("\n" + "=" * 60)
