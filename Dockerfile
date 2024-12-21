# Use an official Python slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    tmux vim cmake python3-dev python3-venv python3-pip imagemagick \
    git git-lfs libopencv-dev python3-opencv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir opencv_contrib_python inky[rpi]==1.5.0 pillow

# Clone and build XNNPACK
RUN git clone https://github.com/google/XNNPACK.git && \
    cd XNNPACK && \
    git checkout 1c8ee1b68f3a3e0847ec3c53c186c5909fa3fbd3 && \
    mkdir build && cd build && \
    cmake -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF .. && \
    cmake --build . --config Release

# Clone and build OnnxStream
RUN git clone https://github.com/vitoplantamura/OnnxStream.git && \
    cd OnnxStream/src && \
    mkdir build && cd build && \
    cmake -DMAX_SPEED=ON -DOS_LLM=OFF -DOS_CUDA=OFF -DXNNPACK_DIR=/app/XNNPACK .. && \
    cmake --build . --config Release

# Download models
RUN mkdir -p /app/models && \
    cd /app/models && \
    git clone --depth=1 https://huggingface.co/AeroX2/stable-diffusion-xl-turbo-1.0-onnxstream

# Set up entrypoint and environment
ENV PATH="/app/venv/bin:$PATH"
CMD ["python", "generate_picture.py", "/output"]
