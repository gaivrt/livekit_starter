import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# 设置日志记录器，用于记录Agent的运行状态和调试信息
logger = logging.getLogger("agent")

# 加载.env文件中的环境变量，包括API密钥等配置
load_dotenv()


class Assistant(Agent):
    """
    语音AI助手类
    
    继承自LiveKit的Agent基类，实现一个友好、乐于助人的语音AI助手。
    该助手具备以下特点：
    - 使用自然语言进行交互
    - 可以调用外部工具（如天气查询）
    - 回答简洁明了，避免复杂格式
    - 具有好奇心、友好性和幽默感
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""你是一个乐于助人的语音AI助手。
            你积极地通过你广泛的知识为用户的问题提供信息和帮助。
            你的回答简洁、切中要点，不使用任何复杂的格式或标点符号。
            你是好奇的、友好的，并且富有幽默感。""",
        )

    # 所有使用@function_tool装饰的函数都会被传递给LLM，当这个agent处于活跃状态时
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """
        天气查询工具函数
        
        使用此工具查询指定地点的当前天气信息。
        如果天气服务不支持该地点，工具会指示这一点，你必须告知用户该地点的天气信息不可用。

        参数:
            location: 要查询天气信息的地点（例如：城市名称）
            
        返回:
            str: 天气信息描述
        """

        logger.info(f"正在查询 {location} 的天气信息")

        # 这里返回模拟的天气信息
        # 在实际应用中，您可以集成真实的天气API服务
        return "晴朗，温度70度。"


def prewarm(proc: JobProcess):
    """
    预热函数
    
    在处理实际任务之前预加载必要的模型和资源，以减少首次响应时间。
    这个函数会在worker进程启动时调用，用于优化性能。
    
    参数:
        proc: 作业进程对象，用于存储预加载的资源
    """
    # 预加载Silero VAD（语音活动检测）模型
    # VAD用于检测音频中的语音段，提高语音识别的准确性
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """
    主入口点函数
    
    这是Agent的主要执行逻辑，负责：
    1. 设置语音AI管道（STT、LLM、TTS）
    2. 配置转话检测和噪音抑制
    3. 启动Agent会话
    4. 处理用户交互
    
    参数:
        ctx: 作业上下文，包含房间信息、配置等
    """
    # 为每个日志条目设置上下文字段，便于调试和监控
    ctx.log_context_fields = {
        "room": ctx.room.name,  # 房间名称
    }

    # 设置语音AI管道，集成OpenAI、Cartesia、Deepgram和LiveKit转话检测器
    session = AgentSession(
        # 可以使用STT、LLM、TTS或实时API的任意组合
        llm=openai.LLM(model="gpt-4o-mini"),                    # 大语言模型：OpenAI GPT-4o-mini
        stt=deepgram.STT(model="nova-3", language="multi"),    # 语音转文本：Deepgram Nova-3，支持多语言
        tts=cartesia.TTS(),                                     # 文本转语音：Cartesia
        # 使用LiveKit的转话检测模型，支持多语言的智能对话转换检测
        turn_detection=MultilingualModel(),
        # 使用预加载的VAD模型进行语音活动检测
        vad=ctx.proc.userdata["vad"],
    )

    # 如果要使用OpenAI实时API，请使用以下会话设置：
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel()
    # )

    # 记录指标数据，用于监控使用情况和成本
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """
        指标收集事件处理函数
        
        当会话收集到新的指标数据时触发，用于实时记录和统计。
        
        参数:
            ev: 包含指标数据的事件对象
        """
        # 记录指标到日志
        metrics.log_metrics(ev.metrics)
        # 收集使用情况统计
        usage_collector.collect(ev.metrics)

    async def log_usage():
        """
        记录使用情况摘要
        
        在会话结束时调用，统计并记录整个会话的资源使用情况，
        包括令牌消耗、API调用次数等。
        """
        summary = usage_collector.get_summary()
        logger.info(f"使用情况统计: {summary}")

    # 添加关闭回调，当会话结束时触发使用情况统计
    ctx.add_shutdown_callback(log_usage)

    # 启动Agent会话
    await session.start(
        agent=Assistant(),  # 使用我们定义的Assistant实例
        room=ctx.room,      # 连接到指定的房间
        room_input_options=RoomInputOptions(
            # LiveKit Cloud增强噪音抑制
            # - 如果是自部署，请省略此参数
            # - 对于电话应用，使用`BVCTelephony`以获得最佳效果
            noise_cancellation=noise_cancellation.BVC(),
        ),
        # 启用转录功能，将语音转换为文本记录
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # 当agent准备就绪时加入房间，开始处理用户交互
    await ctx.connect()


if __name__ == "__main__":
    """
    程序主入口
    
    使用LiveKit CLI运行Agent应用程序。
    配置了入口点函数和预热函数，确保Agent能够正确启动和运行。
    """
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
