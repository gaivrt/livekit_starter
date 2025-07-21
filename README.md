<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Agents 启动项目 - Python版

一个完整的语音AI应用程序启动项目，基于[LiveKit Agents for Python](https://github.com/livekit/agents)构建。

## 项目简介

这个启动项目为您提供了一个功能齐全的语音AI助手框架，可以快速开发和部署语音交互应用。该项目基于LiveKit Agents框架，集成了业界领先的AI服务提供商，为您提供企业级的语音AI解决方案。

## 核心特性

### 🎯 语音AI助手
- 基于[LiveKit语音AI快速入门](https://docs.livekit.io/agents/start/voice-ai/)的完整语音助手实现
- 支持自然语言对话和智能响应
- 内置示例工具函数（天气查询）展示扩展能力

### 🔧 多服务集成
- **大语言模型(LLM)**: [OpenAI GPT-4o-mini](https://docs.livekit.io/agents/integrations/llm/openai/)
- **文本转语音(TTS)**: [Cartesia](https://docs.livekit.io/agents/integrations/tts/cartesia/)
- **语音转文本(STT)**: [Deepgram Nova-3](https://docs.livekit.io/agents/integrations/stt/deepgram/)
- 支持替换为您偏好的[LLM](https://docs.livekit.io/agents/integrations/llm/)、[STT](https://docs.livekit.io/agents/integrations/stt/)和[TTS](https://docs.livekit.io/agents/integrations/tts/)服务
- 支持升级到实时模型，如[OpenAI实时API](https://docs.livekit.io/agents/integrations/realtime/openai)

### 🧪 测试与评估
- 基于LiveKit Agents[测试与评估框架](https://docs.livekit.io/agents/build/testing/)的完整评估套件
- 自动化测试用例覆盖助手行为、工具调用、错误处理等场景
- 支持AI驱动的评估和判断，确保对话质量

### 🎙️ 高级语音功能
- [LiveKit转话检测器](https://docs.livekit.io/agents/build/turns/turn-detector/) - 上下文感知的说话人检测
- 多语言支持，适应不同地区用户
- [Silero VAD](https://docs.livekit.io/agents/build/turns/vad/)语音活动检测
- [LiveKit Cloud增强噪音抑制](https://docs.livekit.io/home/cloud/noise-cancellation/)

### 📊 监控与分析
- 集成[指标和日志记录](https://docs.livekit.io/agents/build/metrics/)
- 实时使用情况统计和性能监控
- 详细的会话分析和用户交互追踪

### 🌐 多平台兼容
- 兼容任何[自定义Web/移动前端](https://docs.livekit.io/agents/start/frontend/)
- 支持[基于SIP的电话系统](https://docs.livekit.io/agents/start/telephony/)集成
- 可部署到生产环境，包含完整的Docker配置

## 开发环境设置

### 1. 克隆项目并安装依赖

```console
cd agent-starter-python
uv sync
```

### 2. 环境配置

复制`.env.example`文件为`.env`并填入必要的配置值：

#### LiveKit配置
- `LIVEKIT_URL`: 使用[LiveKit Cloud](https://cloud.livekit.io/)或[自建服务](https://docs.livekit.io/home/self-hosting/)
- `LIVEKIT_API_KEY`: LiveKit API密钥
- `LIVEKIT_API_SECRET`: LiveKit API密钥

#### AI服务API密钥
- `OPENAI_API_KEY`: [获取OpenAI密钥](https://platform.openai.com/api-keys)或使用您的[首选LLM提供商](https://docs.livekit.io/agents/integrations/llm/)
- `DEEPGRAM_API_KEY`: [获取Deepgram密钥](https://console.deepgram.com/)或使用您的[首选STT提供商](https://docs.livekit.io/agents/integrations/stt/)
- `CARTESIA_API_KEY`: [获取Cartesia密钥](https://play.cartesia.ai/keys)或使用您的[首选TTS提供商](https://docs.livekit.io/agents/integrations/tts/)

#### 快速配置
您可以使用[LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup)自动加载环境变量：

```bash
lk app env -w .env
```

## 运行语音AI助手

### 1. 下载必要模型

首次运行前，需要下载[Silero VAD](https://docs.livekit.io/agents/build/turns/vad/)和[LiveKit转话检测器](https://docs.livekit.io/agents/build/turns/turn-detector/)等模型：

```console
uv run python src/agent.py download-files
```

### 2. 直接在终端测试

在终端中直接与您的语音助手对话：

```console
uv run python src/agent.py console
```

### 3. 开发模式

为前端或电话系统开发时使用：

```console
uv run python src/agent.py dev
```

### 4. 生产部署

生产环境中使用：

```console
uv run python src/agent.py start
```

## 前端与电话集成

快速开始使用我们预构建的前端启动应用，或添加电话支持：

| 平台 | 链接 | 描述 |
|----------|----------|-------------|
| **Web** | [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) | 基于React & Next.js的Web语音AI助手 |
| **iOS/macOS** | [`livekit-examples/agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) | 原生iOS、macOS和visionOS语音AI助手 |
| **Flutter** | [`livekit-examples/agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) | 跨平台语音AI助手应用 |
| **React Native** | [`livekit-examples/voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) | 基于React Native & Expo的原生移动应用 |
| **Android** | [`livekit-examples/agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) | 基于Kotlin & Jetpack Compose的原生Android应用 |
| **Web嵌入组件** | [`livekit-examples/agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) | 可嵌入任何网站的语音AI小部件 |
| **电话系统** | [📚 文档](https://docs.livekit.io/agents/start/telephony/) | 为您的助手添加呼入或呼出功能 |

如需高级定制，请参阅[完整前端指南](https://docs.livekit.io/agents/start/frontend/)。

## 测试与评估

本项目包含基于LiveKit Agents[测试与评估框架](https://docs.livekit.io/agents/build/testing/)的完整评估套件。运行测试：

```console
uv run pytest
```

### 测试覆盖范围
- **友好性测试**: 评估助手的友好问候和互动方式
- **工具功能测试**: 验证天气查询等工具的正确调用和响应
- **错误处理测试**: 测试服务不可用时的优雅降级
- **拒绝测试**: 确保助手正确拒绝不当或有害请求
- **知识边界测试**: 验证助手承认不知道的信息

## 项目结构详解

```
├── src/
│   ├── __init__.py          # Python包初始化文件
│   └── agent.py             # 主要的Agent实现，包含助手逻辑和工具函数
├── tests/
│   └── test_agent.py        # 完整的测试套件，包含各种场景测试
├── .github/
│   └── workflows/           # GitHub Actions CI/CD配置
├── pyproject.toml           # Python项目配置和依赖管理
├── Dockerfile               # 生产环境Docker配置
├── taskfile.yaml            # 任务自动化配置
└── README.md                # 本文档
```

## 基于此模板创建您的项目

当您基于此模板开始自己的项目时，应该：

1. **提交您的`uv.lock`文件**: 此文件当前未被跟踪（仅用于模板），但您应该将其提交到您的仓库以确保可重复的构建和正确的配置管理。（如果您在LiveKit Cloud中运行代理，`livekit.toml`也是如此）

2. **移除git跟踪测试**: 从`.github/workflows/tests.yml`中删除"检查git中未跟踪的文件"步骤，因为您现在希望跟踪这些文件。这些仅用于模板仓库本身的开发目的。

3. **添加您自己的仓库密钥**: 您必须为`OPENAI_API_KEY`或其他LLM提供商[添加密钥](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/using-secrets-in-github-actions)，以便测试可以在CI中运行。

## 生产部署

此项目已准备好用于生产环境，包含可用的`Dockerfile`。要将其部署到LiveKit Cloud或其他环境，请参阅[生产部署指南](https://docs.livekit.io/agents/ops/deployment/)。

### 部署特性
- **Docker支持**: 完整的容器化配置
- **云原生**: 适配各种云平台部署
- **高可用**: 支持负载均衡和自动扩展
- **监控集成**: 内置性能指标和日志记录

## 技术架构

### 核心组件
1. **Agent类**: 继承自LiveKit Agent的主要助手逻辑
2. **工具系统**: 基于装饰器的函数工具，支持LLM调用
3. **会话管理**: 自动处理用户会话和状态管理
4. **多模态支持**: 统一的语音、文本和实时API接口

### 性能优化
- **预热机制**: VAD模型预加载减少启动时间
- **资源管理**: 智能的内存和计算资源使用
- **并发处理**: 支持多用户同时对话
- **缓存策略**: 优化响应时间和成本

## 开发最佳实践

### 代码质量
- **类型提示**: 完整的Python类型注解
- **异步编程**: 基于asyncio的高性能异步架构
- **错误处理**: 优雅的异常处理和用户友好的错误消息
- **日志记录**: 结构化日志便于调试和监控

### 扩展指南
- **自定义工具**: 通过`@function_tool`装饰器添加新功能
- **服务集成**: 轻松替换或添加新的AI服务提供商
- **多语言支持**: 配置不同语言的STT/TTS服务
- **个性化定制**: 调整助手的指令和行为特性

## 许可证

本项目基于MIT许可证 - 详细信息请参阅[LICENSE](LICENSE)文件。

## 支持与社区

- [LiveKit文档](https://docs.livekit.io/)
- [LiveKit Discord社区](https://livekit.io/join-slack)
- [GitHub Issues](https://github.com/livekit/agents/issues)
- [示例项目](https://github.com/livekit-examples)

---

开始构建您的下一个语音AI应用程序！🚀