import torch
from deploy import processed
from src import config


def predict_image(model, device, image_path):

    image = processed.preprocess_image(image_path,config.image_size)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs,dim=1)

    return {
        "class": config.categories[predicted.item()],
        "confidence": float(confidence.item())
    }