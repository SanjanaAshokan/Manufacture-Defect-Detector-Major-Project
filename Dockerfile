FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: exact versions from the PyTorch CPU index,
# all other dependencies come from normal PyPI
RUN pip install --no-cache-dir \
    torch==2.14.1+cpu torchvision==0.29.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p outputs/checkpoints outputs/logs outputs/plots outputs/gradcam_samples

EXPOSE 7860
CMD ["python", "web/app.py"]
