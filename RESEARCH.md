# Voice LLM Bridge - 技术调研报告

## 项目目标

实现实时语音对话系统，连接用户麦克风和 OpenClaw AI 助手：

```
🎤 麦克风 → ASR (语音转文字) → OpenClaw Gateway (WebSocket) → TTS (文字转语音) → 🔊 音箱
```

---

## 1. 模型选型分析

### 1.1 VibeVoice-Realtime (0.5B) - TTS

**模型信息**：
- 参数：0.5B
- HuggingFace: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
- 延迟：~300ms 首音延迟
- 流式输入：支持
- 长文本：支持 ~10 分钟连续生成

**Apple Silicon 兼容性**：
| 推理方式 | 支持度 | 说明 |
|---------|--------|------|
| MLX | ✅ 推荐 | Apple 官方框架，统一内存优化最佳 |
| PyTorch (MPS) | ✅ 支持 | 通过 PyTorch 2.0+ MPS 后端 |
| Transformers | ✅ 支持 | 官方推荐方式 |

**推荐方案**：MLX (与 ASR 统一技术栈，内存共享更高效)

**内存占用**：
- FP16: ~1GB
- 4bit 量化：~0.5GB

---

### 1.2 VibeVoice-ASR (7B) - ASR

**模型信息**：
- 参数：7B
- HuggingFace: https://huggingface.co/microsoft/VibeVoice-ASR
- 能力：60 分钟长语音单次处理、50+ 语言、说话人分离、时间戳
- 输出：结构化转录 (Who/When/What)

**Apple Silicon 兼容性**：
| 推理方式 | 支持度 | 说明 |
|---------|--------|------|
| MLX | ✅ 推荐 | MLX 针对 Apple Silicon 优化，支持 4bit 量化 |
| PyTorch (MPS) | ⚠️ 勉强 | 7B 模型 MPS 可能 OOM |
| Transformers | ⚠️ 慢 | CPU fallback 会很慢 |

**推荐方案**：MLX + 4bit 量化

**内存占用**：
- FP16: ~14GB
- 4bit 量化：~4-5GB (含激活开销)

**替代方案**：Whisper (OpenAI)
- 参数更小 (large-v3: 1.5B)
- Apple Silicon 优化更好
- 但无说话人分离功能

---

## 2. OpenClaw Gateway 集成

### 2.1 WebSocket API 架构

```
Voice Bridge → OpenClaw Gateway (WebSocket) → AI Session → 返回响应
     ↓
   ws://HOST:PORT/?token=TOKEN
```

### 2.2 连接流程

1. **WebSocket 连接**
   ```
   ws://HOST:PORT/?token=TOKEN
   ```

2. **处理 connect.challenge**
   ```javascript
   {
     type: "req", id: "connect-1", method: "connect",
     params: {
       minProtocol: 3, maxProtocol: 3,
       client: { id: "gateway-client", version: "1.0.0", platform: "linux", mode: "backend" },
       role: "operator", scopes: ["operator.admin"],
       auth: { token: TOKEN }
     }
   }
   ```

3. **发送消息**
   ```javascript
   {
     type: "req", id: "chat-1", method: "chat.send",
     params: {
       sessionKey: "main",
       idempotencyKey: uuid(),  // 必须
       message: "用户语音转写的文字"
     }
   }
   ```

4. **接收响应**
   - 监听 `agent` 和 `chat` 事件
   - 检查 `state: "final"` 获取完整响应
   - 或流式处理 `stream: "assistant"` + `data.text`

### 2.3 关键约束

| 约束 | 说明 |
|------|------|
| client.id | 必须用预定义常量 (`gateway-client`) |
| idempotencyKey | 每次 chat.send 必须唯一 |
| 响应格式 | 检查 `state: "final"` 判断完成 |
| 超时 | 建议 120 秒超时 |

---

## 3. 推荐架构

### 3.1 整体架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌─────────────┐
│  麦克风输入  │ ──→ │   ASR 模块    │ ──→ │  OpenClaw Bridge  │ ──→ │   TTS 模块   │
│  (Audio In) │     │ (VibeVoice-  │     │   (WebSocket)   │     │ (VibeVoice- │
│             │     │   ASR/MLX)   │     │                   │     │  Realtime)  │
└─────────────┘     └──────────────┘     └─────────────────┘     └─────────────┘
                           ↓                      ↓                      ↓
                      文字转录              AI 响应文字              音频输出
```

### 3.2 技术栈推荐

| 组件 | 推荐方案 | 备选方案 |
|------|---------|---------|
| ASR 推理 | MLX (Apple Silicon) | PyTorch MPS |
| TTS 推理 | PyTorch MPS | MLX (需转换) |
| 音频 I/O | sounddevice + PyAudio | PyDub |
| OpenClaw 连接 | WebSocket (ws 库) | aiohttp |
| 流程编排 | asyncio | threading |

### 3.3 延迟估算 (MLX + 4bit 量化)

| 阶段 | 预估延迟 | 说明 |
|------|---------|------|
| ASR (7B, MLX 4bit) | 1-3 秒 | 统一内存带宽优势，量化后推理更快 |
| OpenClaw 响应 | 1-3 秒 | 取决于模型负载 |
| TTS (0.5B, MLX 4bit) | 0.3-1 秒 | 流式输出，首音 300ms |
| **总延迟** | **2.3-7 秒** | 可优化到 2-4 秒 (流式 ASR) |

**优化方向**：
1. 流式 ASR (VibeVoice-ASR 支持分块流式)
2. 流式 TTS (VibeVoice-Realtime 原生支持)
3.  Pipeline 并行 (ASR 完成后立即 TTS，不等 OpenClaw 结束)

---

## 4. Mac mini M4 Pro 评估

### 4.1 硬件规格 (假设)

- CPU: M4 Pro (12-14 核)
- GPU: 16-20 核
- 内存：24-48GB 统一内存
- 带宽：~200GB/s

### 4.2 模型运行可行性 (MLX + 4bit 量化)

| 模型 | 内存需求 (量化后) | M4 Pro 可行性 |
|------|------------------|--------------|
| VibeVoice-ASR (7B) | ~4-5GB | ✅ 轻松运行 |
| VibeVoice-Realtime (0.5B) | ~0.5GB | ✅ 轻松运行 |
| **同时运行** | **~5-6GB** | ✅ 24GB 绰绰有余 |

### 4.3 优化建议

1. **统一 MLX 技术栈**：ASR + TTS 都用 MLX，内存共享更高效
2. **4bit 量化**：MLX 原生支持，精度损失小，内存降低 75%
3. **统一内存优势**：M4 Pro 的 24GB 统一内存可同时加载两个模型，无需交替

---

## 5. 风险与替代方案

### 5.1 主要风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| MLX 模型转换复杂 | 高 | 优先用官方 MLX 版本 |
| 内存不足 OOM | 高 | 量化 + 交替加载 |
| 延迟过高 | 中 | 流式处理 + Whisper 替代 |
| OpenClaw WebSocket 不稳定 | 中 | 重试机制 + 心跳 |

### 5.2 替代方案

**方案 A：Whisper + 内置 TTS**
- ASR: Whisper (Apple Silicon 优化好)
- TTS: OpenClaw 内置 TTS 工具
- 优点：成熟稳定
- 缺点：无 VibeVoice 高级功能

**方案 B：云端 ASR/TTS**
- ASR: Azure Speech / Google Cloud Speech
- TTS: Azure TTS / ElevenLabs
- 优点：零本地负载
- 缺点：网络依赖、成本

**方案 C：纯本地 + 简化模型**
- ASR: Whisper small (384M)
- TTS: Piper TTS (更小)
- 优点：低延迟、低内存
- 缺点：音质/准确率略低

---

## 6. 下一步行动

### Phase 1: 原型验证 (1-2 天)
- [ ] 搭建基础 Python 项目结构
- [ ] 实现 OpenClaw WebSocket 客户端
- [ ] 测试 VibeVoice-Realtime TTS (PyTorch MPS)
- [ ] 测试 VibeVoice-ASR 或 Whisper ASR

### Phase 2: 集成开发 (2-3 天)
- [ ] 实现音频输入/输出管道
- [ ] 实现 ASR → OpenClaw → TTS 流程
- [ ] 添加错误处理和重试
- [ ] 性能优化 (延迟/内存)

### Phase 3: 产品化 (2-3 天)
- [ ] Docker Compose 部署
- [ ] 配置文件管理
- [ ] 日志和监控
- [ ] 文档和示例

---

## 7. 项目结构建议

```
voice-llm-bridge/
├── README.md
├── RESEARCH.md (本文件)
├── SPEC.md (详细设计)
├── requirements.txt
├── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── vibevoice_asr.py
│   │   └── whisper_asr.py
│   ├── tts/
│   │   ├── __init__.py
│   │   └── vibevoice_tts.py
│   ├── openclaw/
│   │   ├── __init__.py
│   │   └── gateway_client.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── audio_pipeline.py
│   └── main.py
└── config/
    ├── default.yaml
    └── macos-m4.yaml
```

---

## 8. 结论

**可行性**：✅ 可行，24GB 内存足够

**推荐配置**：
- Mac mini M4 Pro 24GB+ 内存 (统一内存优势)
- ASR: VibeVoice-ASR (MLX + 4bit 量化)
- TTS: VibeVoice-Realtime (MLX + 4bit 量化)
- 延迟：2-7 秒 (流式优化后可到 2-4 秒)
- 内存占用：~5-6GB (两个模型同时加载)

**技术栈**：
- 统一使用 MLX 框架 (Apple Silicon 最优解)
- 4bit 量化降低内存 75%，精度损失可接受
- 统一内存架构让 24GB 实际可用内存远超传统架构

**建议**：先做原型验证 Phase 1，确认 MLX 对两个模型的支持情况和实际性能。
