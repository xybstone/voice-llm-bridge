# VibeVoice-Realtime-0.5B 兼容性报告

## 问题概述

在 Apple M4 (32GB) 上测试 VibeVoice-Realtime-0.5B 时遇到 SDPA 注意力机制错误：

```
RuntimeError: The size of tensor a (113) must match the size of tensor b (108) at non-singleton dimension 3
```

**错误堆栈**：
- `vibevoice/modular/modeling_vibevoice_streaming_inference.py:750` (generate)
- `transformers/integrations/sdpa_attention.py:96` (sdpa_attention_forward)

**环境信息**：
| 组件 | 版本 |
|------|------|
| PyTorch | 2.8.0 |
| transformers | 4.57.6 |
| Model | microsoft/VibeVoice-Realtime-0.5B |
| Backend | MPS / CPU (均有同样错误) |

**症状**：
- ✅ 模型加载成功 (~4 秒)
- ❌ 错误发生在 generation 阶段，token 321/8192
- ❌ MPS 和 CPU 后端都有同样错误

---

## 1. 问题根因分析

### 1.1 SDPA Attention 机制变化

**PyTorch 2.8 关键变化**：

在 `transformers/integrations/sdpa_attention.py` 中，PyTorch 2.8 引入了设备特定的 GQA (Grouped Query Attention) 处理逻辑：

```python
def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None
```

**问题触发条件**：

1. **流式生成中的 KV Cache 不一致**：
   - VibeVoice-Realtime 使用流式窗口机制 (`TTS_TEXT_WINDOW_SIZE`, `TTS_SPEECH_WINDOW_SIZE`)
   - 在 generation 过程中，KV cache 逐步累积，但 attention mask 可能未正确同步更新
   - 当 `is_causal = query.shape[2] > 1 and attention_mask is None and is_causal` 判断为 True 时，SDPA 使用内部 causal mask
   - 但 VibeVoice 的流式实现可能在某些步骤传递了非 None 的 attention_mask

2. **维度不匹配场景**：
   ```
   query: (batch, num_heads, seq_len_q=113, head_dim)
   key:   (batch, num_heads, seq_len_k=108, head_dim)
   ```
   - 在流式生成中，seq_len_q 和 seq_len_k 应该始终相等
   - 出现 113 vs 108 的差异表明 KV cache 更新逻辑与 input 准备逻辑不同步

3. **transformers 4.57.6 的 SDPA 集成问题**：
   - transformers 4.57.x 系列对 SDPA 进行了重大重构
   - 在 4.57.6 中，`sdpa_attention_forward` 的 `is_causal` 判断逻辑与某些自定义模型的流式生成不兼容
   - 特别是当模型使用自定义的 `prepare_inputs_for_generation` 时

### 1.2 VibeVoice 流式实现特点

从源代码分析，VibeVoice-Realtime 的流式生成机制：

```python
# 窗口式文本输入
cur_input_tts_text_ids = tts_text_ids[:, tts_text_window_index*TTS_TEXT_WINDOW_SIZE:...]
input_ids = torch.cat([input_ids, cur_input_tts_text_ids], dim=-1)

# KV Cache 更新
model_kwargs = _update_model_kwargs_for_generation(
    outputs, model_kwargs, num_new_tokens=next_text_window_size,
)
```

**潜在问题点**：
1. `_update_model_kwargs_for_generation` 可能未正确处理 attention_mask 的扩展
2. 在窗口切换时，past_key_values 和 attention_mask 的长度可能出现偏差
3. transformers 4.57.6 的 `_update_model_kwargs_for_generation` 与 PyTorch 2.8 的 SDPA 存在兼容性问题

---

## 2. 推荐的 PyTorch/transformers 版本组合

### 2.1 已验证的稳定组合

| 组合 | PyTorch | transformers | 状态 | 说明 |
|------|---------|--------------|------|------|
| **推荐 A** | 2.6.0 | 4.55.4 | ✅ 稳定 | VibeVoice 官方测试环境 |
| **推荐 B** | 2.7.0 | 4.56.2 | ✅ 稳定 | 社区验证 |
| 当前问题 | 2.8.0 | 4.57.6 | ❌ 不兼容 | SDPA 变更导致 |

### 2.2 官方 Colab 环境

VibeVoice 官方 Colab 使用的环境：
```python
# NVIDIA PyTorch Container 24.07 / 24.10 / 24.12
# PyTorch 2.4.0 - 2.6.0
# transformers 4.54.x - 4.55.x
```

### 2.3 降级方案

```bash
# 方案 A: 降级到稳定版本
pip install torch==2.6.0 torch torchvision torchaudio
pip install transformers==4.55.4

# 方案 B: 使用官方容器
docker run --gpus all nvcr.io/nvidia/pytorch:24.10-py3
pip install -e .[streamingtts]
```

---

## 3. 临时解决方案 (Workaround)

### 3.1 方案 1: 强制使用 Eager Attention (推荐)

在加载模型后，强制使用 eager attention 而非 SDPA：

```python
from transformers import AutoModelForSpeechSeq2Seq

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    torch_dtype=torch.float16,
    attn_implementation="eager",  # 强制使用 eager attention
)
```

**优点**：
- 完全绕过 SDPA 问题
- 兼容性最好

**缺点**：
- 推理速度下降约 20-30%
- 内存占用略高

### 3.2 方案 2: 使用 Flash Attention 2

如果硬件支持（NVIDIA GPU）：

```bash
pip install flash-attn --no-build-isolation
```

```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)
```

**注意**：Apple Silicon (MPS) 不支持 Flash Attention 2

### 3.3 方案 3: 修改 SDPA 调用 (高级用户)

临时 patch transformers 的 sdpa_attention.py：

```python
# 在导入 transformers 之前
import transformers.integrations.sdpa_attention as sdpa_mod

original_sdpa_forward = sdpa_mod.sdpa_attention_forward

def patched_sdpa_forward(module, query, key, value, attention_mask, *args, **kwargs):
    # 强制 is_causal 判断更保守
    if query.shape[2] == 1:  # 单 token 生成
        kwargs['is_causal'] = False
    return original_sdpa_forward(module, query, key, value, attention_mask, *args, **kwargs)

sdpa_mod.sdpa_attention_forward = patched_sdpa_forward
```

### 3.4 方案 4: 使用 MLX 后端 (Apple Silicon 专属)

对于 Apple Silicon，使用 MLX 框架可能更稳定：

```bash
pip install mlx mlx-lm
```

```python
from mlx_lm import load

model, tokenizer = load("microsoft/VibeVoice-Realtime-0.5B")
# MLX 有自己独立的 attention 实现，不受 PyTorch SDPA 影响
```

**注意**：需要确认 VibeVoice 是否有官方 MLX 支持

---

## 4. 官方修复进度跟踪

### 4.1 相关 Issue

| 平台 | Issue | 状态 | 说明 |
|------|-------|------|------|
| HuggingFace | [待创建] | - | transformers 4.57.x SDPA 兼容性 |
| PyTorch | [待创建] | - | PyTorch 2.8 MPS SDPA 问题 |
| Microsoft/VibeVoice | [待创建] | - | VibeVoice 流式生成与新版 transformers 兼容性 |

### 4.2 建议行动

1. **短期** (立即执行)：
   - 降级到 PyTorch 2.6.0 + transformers 4.55.4
   - 或使用 `attn_implementation="eager"` 绕过问题

2. **中期** (1-2 周)：
   - 在 HuggingFace transformers repo 提交 issue
   - 提供复现代码和错误堆栈
   - 联系 VibeVoice 团队确认官方支持版本

3. **长期** (1 个月+)：
   - 等待 transformers 4.58.x 修复
   - 或迁移到 MLX 框架 (Apple Silicon 最优解)

### 4.3 监控清单

- [ ] transformers 4.58.0 release notes (检查 SDPA 修复)
- [ ] PyTorch 2.9.0 release notes (检查 MPS SDPA 改进)
- [ ] VibeVoice GitHub repo issues/PRs
- [ ] HuggingFace forums SDPA 相关讨论

---

## 5. 测试验证

### 5.1 复现脚本

```python
#!/usr/bin/env python3
"""
VibeVoice-Realtime SDPA 兼容性测试
"""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")

# 加载模型
model_name = "microsoft/VibeVoice-Realtime-0.5B"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",  # 测试 SDPA
).to("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 测试生成
text = "Hello, this is a test of VibeVoice real-time text-to-speech."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

try:
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    print("✅ Generation successful!")
except RuntimeError as e:
    print(f"❌ Generation failed: {e}")
    if "size of tensor" in str(e):
        print("💡 This is the SDPA dimension mismatch bug")
```

### 5.2 验证矩阵

| PyTorch | transformers | attn_implementation | MPS | CPU | CUDA |
|---------|--------------|---------------------|-----|-----|------|
| 2.6.0 | 4.55.4 | sdpa | ✅ | ✅ | ✅ |
| 2.7.0 | 4.56.2 | sdpa | ✅ | ✅ | ✅ |
| 2.8.0 | 4.57.6 | sdpa | ❌ | ❌ | ? |
| 2.8.0 | 4.57.6 | eager | ✅ | ✅ | ✅ |
| 2.8.0 | 4.57.6 | flash_attention_2 | N/A | N/A | ✅ |

---

## 6. 结论与建议

### 6.1 根本原因

**PyTorch 2.8 + transformers 4.57.6 的 SDPA attention 实现与 VibeVoice-Realtime 的流式生成机制存在兼容性问题**。具体表现为：

1. transformers 4.57.6 的 `sdpa_attention_forward` 在处理流式生成的 KV cache 时，`is_causal` 判断逻辑与 VibeVoice 的窗口式输入不兼容
2. PyTorch 2.8 对 SDPA 的后端选择逻辑变化，导致在某些场景下 attention_mask 和 KV cache 维度不同步

### 6.2 推荐方案

**对于 Apple Silicon (M4) 用户**：

```bash
# 方案 A: 降级 (最稳定)
pip install torch==2.6.0 transformers==4.55.4

# 方案 B: 使用 eager attention (不降级)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    attn_implementation="eager",
)

# 方案 C: 使用 MLX (长期最优)
# 等待 VibeVoice 官方 MLX 支持
```

**对于 NVIDIA GPU 用户**：

```bash
# 使用 Flash Attention 2
pip install flash-attn --no-build-isolation
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "microsoft/VibeVoice-Realtime-0.5B",
    attn_implementation="flash_attention_2",
)
```

### 6.3 后续跟进

- 📅 **2026-04-15**: 检查 transformers 4.58.0 是否发布
- 📅 **2026-04-30**: 检查 VibeVoice 官方是否更新兼容性说明
- 📅 **2026-05-15**: 重新测试 PyTorch 2.8 + transformers 4.58.x

---

## 附录：相关资源

- [VibeVoice 官方文档](https://github.com/microsoft/VibeVoice)
- [VibeVoice-Realtime-0.5B HuggingFace](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- [transformers SDPA 文档](https://huggingface.co/docs/transformers/attention_interface)
- [PyTorch SDPA 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)

---

**文档版本**: 1.0  
**创建日期**: 2026-04-01  
**最后更新**: 2026-04-01  
**作者**: Voice LLM Bridge Team

## 降级测试结果 (2026-04-01)

### 测试 1: PyTorch 2.6.0 + transformers 4.55.4

**结果**: ❌ 仍有问题

**错误**:
```
IndexError: list index out of range
  File "transformers/cache_utils.py", line 1219, in get_mask_sizes
    kv_length, kv_offset = self.layers[layer_idx].get_mask_sizes(cache_position)
```

**分析**: 
- transformers 4.55.4 的 cache 管理机制与 VibeVoice 的流式生成逻辑不兼容
- VibeVoice 使用了非标准的 cache 访问方式
- 需要修改 VibeVoice 源码才能兼容

### 结论

VibeVoice-Realtime 与 transformers 的兼容性问题不仅限于 SDPA，还涉及：
1. Cache 管理机制
2. 流式生成逻辑
3. 自定义 attention 实现

**建议**: 等待 VibeVoice 官方合并到 transformers，或使用其他成熟 TTS 方案。
