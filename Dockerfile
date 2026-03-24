FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libegl1 \
    libgl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY uv.bin /usr/local/bin/uv

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml .
COPY src/ src/

RUN uv venv && uv pip install -e .

CMD ["bash"]
