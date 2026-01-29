import os
import torch
import shutil
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model1.model1 import Model1


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

TEST_DIR = "data/RAF_aligned_processed/test"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_inference(MODEL_PATH = "models/model1_v0.pth"):

    prepare_output_dir(OUTPUT_DIR)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5],
                             std = [0.5, 0.5, 0.5]),
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform = transform)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    class_names = test_dataset.classes
    num_classes = len(class_names)

    model = Model1(num_classes = num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim = 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    np.save(OUTPUT_DIR / "y_pred.npy", np.array(all_preds))
    np.save(OUTPUT_DIR / "y_true.npy", np.array(all_labels))
    np.save(OUTPUT_DIR / "class_names.npy", np.array(class_names))

    print("[INFO] Evaluation data saved:")
    print(f" - {OUTPUT_DIR / 'y_pred.npy'}")
    print(f" - {OUTPUT_DIR / 'y_true.npy'}")
    print(f" - {OUTPUT_DIR / 'class_names.npy'}")


def prepare_output_dir(output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


