from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_IMPORT_ERROR = None
try:
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms
except Exception as exc:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    Image = None
    transforms = None
    _IMPORT_ERROR = exc

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_card_classifier.pt"
DEFAULT_IMAGE_SIZE = 100


if nn is not None:

    class SmallCardNet(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

else:

    class SmallCardNet:
        def __init__(self, num_classes: int):
            _require_torch()


def build_model(num_classes: int):
    return SmallCardNet(num_classes)


def build_transforms(image_size: int):
    if transforms is None:
        raise RuntimeError("torchvision is not installed") from _IMPORT_ERROR

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
    )


def _pick_device():
    if torch is None:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "Torch and torchvision are required for hand-card classification. "
            "Install with: python3 -m pip install torch torchvision pillow",
        ) from _IMPORT_ERROR


def _opencv_image_to_pil(image):
    """Convert an OpenCV BGR/BGRA array to the RGB PIL input used by the model.

    Screenshots and hand crops in the runtime are decoded by OpenCV.  PIL
    interprets a three-channel array as RGB, so passing the crop through
    unchanged silently swaps red and blue and changes classifier predictions.
    Grayscale and already-PIL inputs remain supported for callers outside the
    runtime path.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for hand-card classification") from _IMPORT_ERROR
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = image
    if getattr(array, "ndim", 0) == 3 and array.shape[2] >= 3:
        array = array[..., :3][..., ::-1]
    # PIL does not accept negative-stride views on all supported versions.
    return Image.fromarray(array.copy() if hasattr(array, "copy") else array)


@dataclass
class HandClassifier:
    model: "torch.nn.Module"
    class_names: list[str]
    image_size: int
    device: "torch.device"
    transform: "transforms.Compose"

    def predict(self, image):
        if Image is None:
            raise RuntimeError("Pillow is required for hand-card classification") from _IMPORT_ERROR
        if torch is None:
            raise RuntimeError("Torch is required for hand-card classification") from _IMPORT_ERROR

        pil_image = _opencv_image_to_pil(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
        return self.class_names[int(idx.item())], float(conf.item())

    def predict_topk(self, image, k: int = 3) -> list[tuple[str, float]]:
        if Image is None:
            raise RuntimeError("Pillow is required for hand-card classification") from _IMPORT_ERROR
        if torch is None:
            raise RuntimeError("Torch is required for hand-card classification") from _IMPORT_ERROR

        k = max(1, int(k))
        pil_image = _opencv_image_to_pil(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            topk = min(k, probs.shape[1])
            confs, indices = torch.topk(probs, topk, dim=1)
        results = []
        for conf, idx in zip(confs[0].tolist(), indices[0].tolist()):
            results.append((self.class_names[int(idx)], float(conf)))
        return results


def load_hand_classifier(model_path: str | Path = DEFAULT_MODEL_PATH, device: str | None = None) -> HandClassifier:
    _require_torch()

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Hand-card classifier model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)

    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device_name = device or _pick_device()
    device_obj = torch.device(device_name)
    model.to(device_obj)

    return HandClassifier(
        model=model,
        class_names=class_names,
        image_size=image_size,
        device=device_obj,
        transform=build_transforms(image_size),
    )
