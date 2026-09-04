# Chat R' Us trainer: teaches Chroma what she looks like from the pictures you
# uploaded, as a small LoRA file the picture worker then loads on every picture.
# Runs as a RunPod serverless endpoint on the same network volume as the
# picture worker (the LoRA lands in /runpod-volume/models/loras/).
#
# Build on RunPod: Serverless -> New Endpoint -> GitHub -> a repo holding this
# folder. See README.md next to it.

FROM ostris/aitoolkit:latest

ENV HF_HOME=/runpod-volume/hf-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf-cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir runpod pyyaml hf_transfer

COPY handler.py /handler.py

# the base image starts its web UI; a serverless worker runs the handler instead
ENTRYPOINT []
CMD ["python", "-u", "/handler.py"]
