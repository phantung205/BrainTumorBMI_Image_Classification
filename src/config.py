import os
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(base_dir, "data", "raw")

image_size = 224
batch_size = 8
learning_rate = 1e-3
momentum = 0.9
epochs = 100


categories = ["notumor","glioma","meningioma","pituitary"]
num_class = len(categories)

report_dir = os.path.join(base_dir,"reports")
path_tensorboard = os.path.join(report_dir,"tensorboard")


model_dir = os.path.join(base_dir,"trained_models")