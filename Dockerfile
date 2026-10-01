# Hugging Face Space (Docker, free CPU). Serves the site + detection API on port 7860.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH=/home/user/app/src \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "ultralytics>=8.3" "opencv-python-headless>=4.9" "numpy>=1.26"

# Weights are not in git; fetch them at build time.
RUN mkdir models \
    && curl -fL -o models/yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt \
    && curl -fL -o models/yoloe-v8l-seg-pf.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg-pf.pt

COPY --chown=user src ./src
COPY --chown=user public ./public

EXPOSE 7860
CMD ["python", "-m", "vision_assist.api", "--host", "0.0.0.0", "--port", "7860", "--no-browser", \
     "--model", "models/yolo11m.pt", "--lens-model", "models/yoloe-v8l-seg-pf.pt"]
