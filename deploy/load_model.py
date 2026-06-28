import torch
from src.model import BrainTumorMRICNN


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    model = BrainTumorMRICNN().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model"])

    model.eval()

    return model, device