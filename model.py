from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


LABELS = {
    "no_drone": None,
    "unknown_drone": "unknown",
    "dji_mini_4_pro": "DJI Mini 4 Pro",
    "dji_air_3": "DJI Air 3",
    "dji_mavic_3_pro": "DJI Mavic 3 Pro",
    "dji_air_2s": "DJI Air 2S",
    "dji_air_3s": "DJI Air 3S",
    "dji_mavic_3": "DJI Mavic 3",
    "dji_mavic_3_classic": "DJI Mavic 3 Classic",
    "dji_mavic_2_pro": "DJI Mavic 2 Pro",
    "dji_avata_2": "DJI Avata 2",
    "dji_mini_3_pro": "DJI Mini 3 Pro",
}

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def build_model(num_classes: int, *, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
