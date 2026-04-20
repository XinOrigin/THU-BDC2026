FROM python:3.12-slim-bookworm

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 【修改点 1】不再复制旧的 uv.lock 账本，只复制我们改好的 pyproject.toml
COPY pyproject.toml ./

# 强制使用清华源
ENV UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

# 【修改点 2】去掉 --frozen 限制，让它根据清华源重新生成下载链接！
RUN uv sync

COPY . .

# Set environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

CMD ["sleep", "infinity"]