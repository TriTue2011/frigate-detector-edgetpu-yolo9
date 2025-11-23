FROM python:3.9-slim-bullseye

# 1. Install system dependencies required for compiling pycocotools and opencv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Add Google Coral Debian package repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# 3. Install libedgetpu (The userspace driver for Coral)
RUN apt-get update && apt-get install -y \
    libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
# tflite-runtime matches the coral libraries
RUN pip install --no-cache-dir \
    tflite-runtime==2.9.1 \
    "numpy<2" \
    opencv-python-headless \
    pydantic \
    pycocotools

# 5. Set working directory
WORKDIR /app

# (Optional) Copy scripts if you want them baked in, 
# but usually mounting them is better for editing.
COPY . /app

ENTRYPOINT ["python", "benchmark.py"]
