"""
LiveKit Agents 语音AI助手模块

这个包提供了一个完整的语音AI助手实现，基于LiveKit Agents框架构建。
主要功能包括：

- 语音到文本(STT)处理
- 大语言模型(LLM)对话处理  
- 文本到语音(TTS)合成
- 工具函数调用支持
- 多语言转话检测
- 噪音抑制和语音增强

使用方法：
    from agent import Assistant
    
    # 创建助手实例
    assistant = Assistant()

模块结构：
    - agent.py: 主要的Assistant类实现和入口点函数
    - __init__.py: 包初始化文件（本文件）

支持的AI服务：
    - OpenAI GPT系列模型 (LLM)
    - Deepgram Nova-3 (STT)
    - Cartesia (TTS)
    - Silero VAD (语音活动检测)

作者: LiveKit团队
许可证: MIT
版本: 1.0.0
"""
