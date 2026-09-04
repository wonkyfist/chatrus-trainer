# Chat R' Us trainer: teaches Chroma what she looks like from the pictures you
# uploaded, as a small LoRA file the picture worker then loads on every picture.
# Runs as a RunPod serverless endpoint on the same network volume as the
# picture worker (the LoRA lands in /runpod-volume/models/loras/).
#
# Built the way ai-toolkit's README installs it (its own image does not allow
# extra packages to be added). Build on RunPod: Serverless -> New Endpoint ->
# GitHub -> a repo holding this folder. See README.md next to it.

FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf-cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recurse-submodules https://github.com/ostris/ai-toolkit.git /app/ai-toolkit
WORKDIR /app/ai-toolkit

# torch wheels carry their own CUDA runtime; the cu128 build runs on every current RunPod host
# (ai-toolkit's README pins 2.13, which only ships for CUDA 13.0; 2.11 is the newest cu128 build)
RUN pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt \
    && pip install runpod pyyaml hf_transfer

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
