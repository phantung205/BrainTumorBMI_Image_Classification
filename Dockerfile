FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /BrainTumor

RUN pip install scikit-learn tensorboard

COPY src ./src

CMD ["python", "-m", "src.train"]