#!/usr/bin/env python3
"""VibeVoice TTS 测试 - 强制使用 CPU"""

import os
import sys
import time

# 强制使用 CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['DEVICE'] = 'cpu'

model_path = os.path.expanduser("~/models/VibeVoice-Realtime-0.5B")
txt_path = os.path.expanduser("~/VibeVoice/demo/text_examples/1p_vibevoice.txt")
voice_path = os.path.expanduser("~/VibeVoice/demo/voices/streaming_model/en-Carter_man.pt")

print("=" * 60)
print("VibeVoice-Realtime TTS 测试 (CPU)")
print("=" * 60)

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
import torch

print(f"\n📂 模型路径：{model_path}")
print(f"📄 文本路径：{txt_path}")
print(f"🎤 语音路径：{voice_path}")

# 加载模型
print("\n🔧 加载模型...")
start = time.time()

device = "cpu"
dtype = torch.float32

processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map=device,
    attn_implementation="sdpa"
)

load_time = time.time() - start
print(f"✅ 模型加载完成：{load_time:.2f}秒")
print(f"   设备：{device}")
print(f"   精度：{dtype}")

# 读取文本
print(f"\n📖 读取文本...")
with open(txt_path) as f:
    text = f.read().strip()
print(f"   文本长度：{len(text)} 字符")
print(f"   内容：{text[:100]}...")

# 加载语音
print(f"\n🎤 加载语音...")
voice_embedding = torch.load(voice_path, map_location=device, weights_only=False)
print(f"   语音嵌入形状：{voice_embedding.shape if hasattr(voice_embedding, 'shape') else type(voice_embedding)}")

# 生成
print(f"\n🔊 开始生成...")
start = time.time()

try:
    outputs = model.generate(
        text=[text],
        voice_embedding=[voice_embedding],
        cfg_scale=1.5,
        max_new_tokens=4096,
        do_sample=False,
    )
    
    gen_time = time.time() - start
    print(f"✅ 生成完成：{gen_time:.2f}秒")
    
    # 保存音频
    output_path = "/tmp/vibevoice_output.wav"
    from vibevoice.utils.audio_utils import save_audio
    save_audio(outputs.audios[0].cpu().numpy(), output_path, sampling_rate=16000)
    print(f"💾 音频已保存：{output_path}")
    
    import subprocess
    subprocess.run(["ls", "-lh", output_path])
    
except Exception as e:
    print(f"❌ 生成失败：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
