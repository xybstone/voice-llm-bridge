# Voice LLM Bridge - 待办事项

## 等待官方修复

### VibeVoice 兼容性等待清单

**问题**: VibeVoice-Realtime-0.5B 与当前 PyTorch/transformers 版本不兼容

**已测试版本**:
- ❌ PyTorch 2.8.0 + transformers 4.57.6 → SDPA tensor 尺寸不匹配
- ❌ PyTorch 2.6.0 + transformers 4.55.4 → Cache 管理机制冲突

**等待事项**:
- [ ] VibeVoice 合并到 transformers 官方
- [ ] 微软发布兼容性修复版本
- [ ] HuggingFace 更新 transformers 支持 vibevoice_streaming 模型类型

**下周行动** (2026-04-08):
- [ ] 检查 VibeVoice GitHub 是否有更新
- [ ] 检查 transformers release notes
- [ ] 重新测试最新版本的兼容性

**相关 Issue**:
- https://github.com/microsoft/VibeVoice/issues (待创建)
- https://github.com/huggingface/transformers/issues (待创建)
- https://github.com/xybstone/voice-llm-bridge/issues/1

---
*最后更新：2026-04-01*
