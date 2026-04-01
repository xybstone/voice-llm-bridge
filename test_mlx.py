#!/usr/bin/env python3
"""
VibeVoice MLX 测试脚本 - 验证 MLX 在 M4 上的运行
"""

import sys
import platform

print("=" * 60)
print("VibeVoice MLX 环境测试")
print("=" * 60)

# 1. 系统信息
print(f"\n📱 系统信息:")
print(f"   Python: {sys.version}")
print(f"   Platform: {platform.platform()}")
print(f"   Machine: {platform.machine()}")

# 2. 测试 MLX
print(f"\n🔧 测试 MLX...")
try:
    import mlx.core as mx
    print(f"   ✅ MLX 版本：{mx.__version__}")
    
    # 测试 Metal 后端
    print(f"\n💻 测试 Metal 后端...")
    device = mx.metal.device_info()
    print(f"   Metal 可用：✅")
    print(f"   设备：{device.get('name', 'Unknown')}")
    print(f"   内存池大小：{device.get('memory_size', 0) / (1024**3):.1f} GB")
    
    # 简单计算测试
    print(f"\n🧮 计算测试...")
    a = mx.random.uniform(shape=(100, 100))
    b = mx.random.uniform(shape=(100, 100))
    c = mx.matmul(a, b)
    mx.eval(c)
    print(f"   ✅ 100x100 矩阵乘法成功")
    
except Exception as e:
    print(f"   ❌ MLX 错误：{e}")
    sys.exit(1)

# 3. 测试 mlx-lm
print(f"\n📚 测试 mlx-lm...")
try:
    import mlx_lm
    print(f"   ✅ mlx-lm 已安装")
    print(f"   版本：{mlx_lm.__version__ if hasattr(mlx_lm, '__version__') else 'Unknown'}")
except Exception as e:
    print(f"   ❌ mlx-lm 错误：{e}")

# 4. 测试 transformers
print(f"\n🤗 测试 transformers...")
try:
    import transformers
    print(f"   ✅ transformers 版本：{transformers.__version__}")
except Exception as e:
    print(f"   ❌ transformers 错误：{e}")

# 5. 测试 huggingface-hub
print(f"\n🦊 测试 huggingface-hub...")
try:
    from huggingface_hub import list_repo_files
    print(f"   ✅ huggingface-hub 已安装")
    
    # 测试连接 (列出 VibeVoice-Realtime 的文件)
    print(f"\n📥 测试 HuggingFace 连接...")
    try:
        files = list_repo_files("microsoft/VibeVoice-Realtime-0.5B", repo_type="model")
        print(f"   ✅ 可访问 microsoft/VibeVoice-Realtime-0.5B")
        print(f"   文件数：{len(files)}")
        print(f"   前 5 个文件：{files[:5]}")
    except Exception as e:
        print(f"   ⚠️ HuggingFace 连接失败：{e}")
        print(f"   可能需要配置代理或 HF_TOKEN")
        
except Exception as e:
    print(f"   ❌ huggingface-hub 错误：{e}")

print("\n" + "=" * 60)
print("✅ 环境测试完成!")
print("=" * 60)

print(f"\n💡 下一步:")
print(f"   1. 下载 VibeVoice-Realtime-0.5B 模型")
print(f"   2. 测试 TTS 推理")
print(f"   3. 下载 VibeVoice-ASR 模型")
print(f"   4. 测试 ASR 推理")
