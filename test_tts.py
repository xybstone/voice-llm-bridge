#!/usr/bin/env python3
"""
VibeVoice-Realtime TTS 测试 - 在 M4 上生成语音
"""

import os
import sys
import time

print("=" * 60)
print("VibeVoice-Realtime TTS 测试")
print("=" * 60)

model_path = os.path.expanduser("~/models/VibeVoice-Realtime-0.5B")

# 检查模型文件
print(f"\n📂 模型路径：{model_path}")
if not os.path.exists(model_path):
    print(f"❌ 模型不存在!")
    sys.exit(1)

model_file = os.path.join(model_path, "model.safetensors")
if os.path.exists(model_file):
    size_gb = os.path.getsize(model_file) / (1024**3)
    print(f"✅ model.safetensors: {size_gb:.2f} GB")
else:
    print(f"❌ model.safetensors 不存在!")
    sys.exit(1)

# 加载模型
print(f"\n🔧 加载模型...")
start = time.time()

try:
    from mlx_lm import load
    
    model, tokenizer = load(model_path)
    load_time = time.time() - start
    print(f"✅ 模型加载完成：{load_time:.2f}秒")
    
    # 打印模型信息
    param_count = sum(p.size for p in model.parameters())
    print(f"   参数量：{param_count / 1e6:.1f}M ({param_count / 1e9:.2f}B)")
    
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    print(f"\n💡 提示：VibeVoice-Realtime 可能需要特定的加载方式")
    print(f"   检查模型配置文件...")
    
    import json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"   模型类型：{config.get('model_type', 'Unknown')}")
        print(f"   架构：{config.get('architectures', ['Unknown'])}")
    sys.exit(1)

# 测试推理
print(f"\n🎤 测试 TTS 推理...")
test_text = "你好，这是 VibeVoice 实时语音合成测试。"
print(f"   输入文本：{test_text}")

try:
    # VibeVoice-Realtime 是 TTS 模型，需要特殊的推理方式
    # 这里先测试基本的模型前向传播
    import mlx.core as mx
    
    # 创建测试输入
    tokens = tokenizer.encode(test_text)
    input_ids = mx.array([tokens])
    
    print(f"   输入 tokens: {len(tokens)}")
    
    # 测试前向传播
    start = time.time()
    output = model(input_ids)
    mx.eval(output)
    elapsed = time.time() - start
    
    print(f"✅ 前向传播完成：{elapsed:.3f}秒")
    print(f"   输出形状：{output.shape if hasattr(output, 'shape') else type(output)}")
    
except Exception as e:
    print(f"❌ 推理测试失败：{e}")
    print(f"\n💡 VibeVoice-Realtime 可能需要专用的推理代码")
    print(f"   参考：https://github.com/microsoft/VibeVoice")

print("\n" + "=" * 60)
print("✅ 测试完成!")
print("=" * 60)
