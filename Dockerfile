FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /BrainTumor

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

RUN pip install scikit-learn tensorboard opencv-python-headless flask

COPY src ./src
COPY deploy ./deploy
COPY templates ./templates
COPY app.py ./app.py

EXPOSE 5000

CMD ["python", "app.py"]