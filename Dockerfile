# Use a base image with Python and CUDA
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface-cli for downloading models
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# Set environment variables
ENV MODEL_BASE=/app/weights
ENV PYTHONPATH=/app
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create weights directory
RUN mkdir -p ${MODEL_BASE}

# Download pretrained models
# It's important to change to the correct directory before downloading
RUN cd ${MODEL_BASE} && huggingface-cli download tencent/HunyuanVideo-Avatar --local-dir ./ --exclude "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"

# Copy the current directory contents into the container at /app
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make scripts executable (if any, good practice)
# RUN chmod +x /app/scripts/*.sh

# Expose any necessary port (if running a web service directly in Docker, not needed for RunPod serverless worker)
# EXPOSE 8000

# Define the default command to run the handler (RunPod will likely override this)
# CMD ["python", "handler.py"]
