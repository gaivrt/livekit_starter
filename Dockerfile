# 这个示例Dockerfile为LiveKit语音AI助手创建一个生产就绪的容器
# syntax=docker/dockerfile:1

# 使用官方UV Python基础镜像，搭载Python 3.11运行在Debian Bookworm上
# UV是一个快速的Python包管理器，比pip提供更好的性能
# 我们使用slim变体来保持镜像大小更小，同时仍然拥有必要的工具
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# 防止Python缓冲stdout和stderr，避免应用崩溃时
# 由于缓冲而没有发出任何日志的情况
ENV PYTHONUNBUFFERED=1

# 创建一个非特权用户，应用程序将在该用户下运行
# 参见：https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# 安装构建Python包时需要的依赖项，特别是有原生扩展的包
# gcc: 构建带有C扩展的Python包所需的C编译器
# python3-dev: 编译时需要的Python开发头文件
# 安装后清理apt缓存以保持镜像大小较小
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录为用户的家目录
# 这是我们的应用程序代码将存放的位置
WORKDIR /home/appuser

# 将所有应用程序文件复制到容器中
# 这包括源代码、配置文件和依赖项规范
# （排除.dockerignore中指定的文件）
COPY . .

# 将所有应用文件的所有权更改为非特权用户
# 这确保应用程序可以根据需要读取/写入文件
RUN chown -R appuser:appuser /home/appuser

# 切换到非特权用户进行所有后续操作
# 这通过不以root身份运行来提高安全性
USER appuser

# 为用户创建缓存目录
# 这被UV和Python用于缓存包和字节码
RUN mkdir -p /home/appuser/.cache

# 使用UV的锁定文件安装Python依赖项
# --locked确保我们使用uv.lock中的确切版本进行可重现的构建
# 这会创建一个虚拟环境并安装所有依赖项
# 确保您的uv.lock文件已检入以保持跨环境的一致性
RUN uv sync --locked

# 预下载代理需要的任何ML模型或文件
# 这确保容器可以立即运行，无需在运行时下载
# 依赖项，这提高了启动时间和可靠性
RUN uv run src/agent.py download-files

# 暴露健康检查端口
# 这允许Docker和编排系统检查容器是否健康
EXPOSE 8081

# 使用UV运行应用程序
# UV将激活虚拟环境并运行代理
# "start"命令告诉worker连接到LiveKit并开始等待作业
CMD ["uv", "run", "src/agent.py", "start"]
