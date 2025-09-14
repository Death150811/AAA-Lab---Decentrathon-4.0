# AAA-Lab---Decentrathon-4.0
# car_ai.py
"""
Unified script for Car Condition AI:
- Train:   python car_ai.py --mode train --data_dir data --epochs 5
- Predict: python car_ai.py --mode predict --image path/to/image.jpg --model models/model.pth
- Serve:   python car_ai.py --mode serve --model models/model.pth

Requirements:
pip install torch torchvision pillow gradio scikit-learn

Project structure expected:
CarAI/
  data/
    train/    <-- class subfolders, e.g. clean/, dirty/, broken/...
    val/      <-- optional validation subfolders, same classes
  models/
"""

import os
import argparse
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# Optional: Gradio for serve mode
try:
    import gradio as gr
except Exception:
    gr = None

# -----------------------
# Helpers / Config
# -----------------------
DEFAULT_MODEL_PATH = "models/model.pth"
CLASS_FILE = "models/classes.json"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

# -----------------------
# Training
# -----------------------
def train(data_dir: str, model_out: str = DEFAULT_MODEL_PATH, epochs: int = 5, batch_size: int = 16, lr: float = 1e-4):
    device = get_device()
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    # If data contains train/val subfolders use them, otherwise use root with class subfolders
    train_root = data_dir / "train" if (data_dir / "train").exists() else data_dir
    val_root = data_dir / "val" if (data_dir / "val").exists() else None

    print("Train root:", train_root)
    print("Val root :", val_root if val_root else "(no val folder, will use train only)")

    transform_train = build_transforms(train=True)
    transform_val = build_transforms(train=False)

    train_dataset = datasets.ImageFolder(str(train_root), transform=transform_train)
    if val_root:
        val_dataset = datasets.ImageFolder(str(val_root), transform=transform_val)
    else:
        val_dataset = None

    classes = train_dataset.classes
    num_classes = len(classes)
    print("Detected classes:", classes)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes (subfolders inside data/train or data).")

    # loaders
    num_workers = 0 if os.name == "nt" else 4  # Windows safer with 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset else None

    # model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        # validation
        val_loss = None
        val_acc = None
        if val_loader:
            model.eval()
            vloss = 0.0
            vcorrect = 0
            vtotal = 0
            with torch.no_grad():
                for vinputs, vlabels in val_loader:
                    vinputs = vinputs.to(device)
                    vlabels = vlabels.to(device)
                    vouts = model(vinputs)
                    vloss_batch = criterion(vouts, vlabels)
                    vpreds = vouts.argmax(dim=1)
                    vloss += vloss_batch.item() * vinputs.size(0)
                    vcorrect += torch.sum(vpreds == vlabels).item()
                    vtotal += vinputs.size(0)
            val_loss = vloss / vtotal
            val_acc = vcorrect / vtotal
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # save best
                os.makedirs(Path(model_out).parent, exist_ok=True)
                torch.save(model.state_dict(), model_out)
                print(f"Saved best model to {model_out} (val_acc={val_acc:.4f})")

        print(f"Epoch [{epoch+1}/{epochs}] train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} "
              + (f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}" if val_loss is not None else "no val"))

        print("Epoch time:", round(time.time()-t0, 1), "s")

    # final save (if no val used)
    os.makedirs(Path(model_out).parent, exist_ok=True)
    if not val_loader:
        torch.save(model.state_dict(), model_out)
        print(f"Saved model to {model_out}")

    # save classes
    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print("Classes saved to", CLASS_FILE)
    print("Training finished.")

# -----------------------
# Predict
# -----------------------
def load_model(model_path: str):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(CLASS_FILE).exists():
        raise FileNotFoundError(f"Classes file not found: {CLASS_FILE}. Train first to create classes.json")

    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        classes = json.load(f)

    device = get_device()
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    return model, classes

def predict_image(model, classes, pil_image):
    # pil_image: PIL.Image
    transform = build_transforms(train=False)
    x = transform(pil_image).unsqueeze(0)
    device = get_device()
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(out.argmax(dim=1).cpu().numpy()[0])
    return {"label": classes[pred_idx], "index": pred_idx, "probs": probs.tolist()}

# -----------------------
# Gradio serve
# -----------------------
def serve(model_path: str):
    if gr is None:
        raise RuntimeError("gradio is not installed. Install with: pip install gradio")

    model, classes = load_model(model_path)

    def infer_fn(img):
        # gradio passes PIL image when type="pil"
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
        else:
            pil = Image.fromarray(img).convert("RGB")
        res = predict_image(model, classes, pil)
        label = res["label"]
        prob = max(res["probs"])
        return f"{label} ({prob*100:.1f}%)"

    iface = gr.Interface(fn=infer_fn,
                         inputs=gr.Image(type="pil"),
                         outputs="text",
                         title="Car Condition Classifier",
                         description="Upload a photo of a car. Model predicts class (e.g., clean, dirty, broken...)")
    iface.launch()

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train","predict","serve"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir, model_out=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.mode == "predict":
        if not args.image:
            print("Provide --image path/to/image.jpg")
            return
        model, classes = load_model(args.model)
        pil = Image.open(args.image).convert("RGB")
        res = predict_image(model, classes, pil)
        probs_str = ", ".join([f"{c}:{p*100:.1f}%" for c,p in zip(classes, res["probs"])])
        print("Prediction:", res["label"])
        print("Probs:", probs_str)
    elif args.mode == "serve":
        serve(args.model)

if __name__ == "__main__":
    main()
