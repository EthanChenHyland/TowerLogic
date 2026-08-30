from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from towerlogic.detection.hand_classifier import build_model


SPECIAL_CLASS_MAP = {
    "the_log": "log",
    "barbarian_barrel": "barb_barrel",
    "barbarian_hut": "barb_hut",
    "goblin_hut": "goblin_hut",
    "goblin_cage": "goblin_cage",
    "goblin_barrel": "goblin_barrel",
    "goblin_drill": "goblin_drill",
    "goblin_gang": "goblin_gang",
    "goblin_giant": "goblin_giant",
    "skeleton_army": "skeleton_army",
    "skeleton_barrel": "skeleton_barrel",
    "skeleton_dragons": "skeleton_dragons",
    "skeleton_king": "skeleton_king",
    "wall_breakers": "wall_breakers",
    "royal_recruits": "royal_recruits",
    "royal_hogs": "royal_hogs",
    "royal_ghost": "royal_ghost",
    "royal_giant": "royal_giant",
    "three_musketeers": "three_musketeers",
    "mega_knight": "mega_knight",
    "mega_minion": "mega_minion",
    "ice_golem": "ice_golem",
    "ice_spirit": "ice_spirit",
    "fire_spirit": "fire_spirit",
    "heal_spirit": "heal_spirit",
    "elixir_golem": "elixir_golem",
    "elixir_collector": "elixir_collector",
    "electro_giant": "electro_giant",
    "electro_wizard": "electro_wizard",
    "electro_dragon": "electro_dragon",
    "electro_spirit": "electro_spirit",
    "archer_queen": "archer_queen",
    "little_prince": "little_prince",
    "magic_archer": "magic_archer",
    "dart_goblin": "dart_goblin",
    "battle_healer": "battle_healer",
    "battle_ram": "battle_ram",
    "barbarian_barrel": "barb_barrel",
    "giant_skeleton": "giant_skeleton",
    "giant_snowball": "snowball",
    "ram_rider": "ram_rider",
    "lava_hound": "lava_hound",
    "inferno_dragon": "inferno_dragon",
    "inferno_tower": "inferno_tower",
    "baby_dragon": "baby_dragon",
    "night_witch": "night_witch",
    "firecracker": "fire_cracker",
    "evo_fire_cracker": "evo_fire_cracker",
    "archers": "archers",
    "hog_rider": "hog",
    "x_bow": "xbow",
    "xbow": "xbow",
    "lighhning": "lightning",
}


def normalize_class_name(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return SPECIAL_CLASS_MAP.get(cleaned, cleaned)


def pick_device(device: str | None) -> str:
    if device:
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_transforms(image_size: int, train: bool):
    base = [transforms.Resize((image_size, image_size))]
    if train:
        base += [
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(base)


def resolve_split(root: Path, name: str) -> Path:
    for option in [name, "valid" if name == "val" else "val"]:
        candidate = root / option
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Split '{name}' not found under {root}")


class RemappedImageFolder(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.ImageFolder, class_to_idx: dict[str, int]):
        self.dataset = dataset
        self.class_to_idx = class_to_idx
        self.samples = []
        for path, original_idx in dataset.samples:
            class_name = dataset.classes[original_idx]
            if class_name not in class_to_idx:
                continue
            self.samples.append((path, class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.dataset.loader(path)
        if self.dataset.transform is not None:
            sample = self.dataset.transform(sample)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return sample, target


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += float(loss.item()) * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args: argparse.Namespace):
    data_root = Path(args.data_dir).expanduser().resolve()
    train_dir = resolve_split(data_root, "train")
    val_dir = resolve_split(data_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(args.image_size, train=True))
    val_raw = datasets.ImageFolder(val_dir, transform=build_transforms(args.image_size, train=False))

    raw_class_names = train_ds.classes
    class_names = [normalize_class_name(name) for name in raw_class_names]

    if len(set(class_names)) != len(class_names):
        duplicates = {name for name in class_names if class_names.count(name) > 1}
        raise ValueError(f"Duplicate normalized class names detected: {sorted(duplicates)}")

    device_name = pick_device(args.device)
    device = torch.device(device_name)

    model = build_model(len(class_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type != "cpu"),
    )
    val_ds = RemappedImageFolder(val_raw, train_ds.class_to_idx)
    if len(val_ds) == 0:
        raise ValueError("Validation set has no samples after class filtering")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type != "cpu"),
    )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}",
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            checkpoint = {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "raw_class_names": raw_class_names,
                "image_size": args.image_size,
                "model": "small_card_net_v1",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.output)

    mapping_path = args.output.with_suffix(".class_map.json")
    mapping_path.write_text(
        json.dumps(
            {
                "raw_class_names": raw_class_names,
                "class_names": class_names,
            },
            indent=2,
        )
    )
    print(f"Saved best model to {args.output} (val_acc={best_acc:.4f})")
    print(f"Saved class map to {mapping_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train hand-card classifier")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the Roboflow folder dataset (containing train/valid/test)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "towerlogic" / "models" / "hand_card_classifier.pt"),
        help="Output path for trained model",
    )
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output = Path(args.output).expanduser().resolve()
    train(args)
