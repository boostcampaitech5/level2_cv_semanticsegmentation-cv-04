# python native
import os
import json
import random
from argparse import ArgumentParser

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# lightweight
import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig

CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def get_filename(IMAGE_ROOT):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    return pngs


class XRayInferenceDataset(Dataset):
    def __init__(self, data_path, transforms=None) -> None:
        super().__init__()

        self.filenames = np.array(sorted(get_filename(data_path)))
        self.transforms = transforms
        self.data_path = data_path

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.data_path, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first

        image = torch.from_numpy(image).float()

        return image, image_name


class InferenceDataModule(L.LightningDataModule):
    def __init__(self, test_dataset) -> None:
        super().__init__()
        self.test_dataset = test_dataset

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            drop_last=False,
        )


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def save_result(exp_name, rles, filename_and_class):
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )

    SAVE_PATH = f"/opt/ml/input/code/smp/result/{exp_name}"
    os.makedirs(SAVE_PATH, exist_ok=True)

    print(f"result is saving in {SAVE_PATH}")
    df.to_csv(f"{SAVE_PATH}/{exp_name}.csv", index=False)


# config에서 모델 정보를 가지고 와야함
@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    tf = A.Resize(512, 512)
    data_path = "/opt/ml/input/data/test/DCM"
    test_dataset = XRayInferenceDataset(data_path, transforms=tf)
    datamodule = InferenceDataModule(test_dataset)
    model = instantiate(cfg["model"]["model"])

    exp_name = cfg["exp_name"]
    ckpt_path = f"/opt/ml/input/code/smp/checkpoints/{exp_name}/best.ckpt"
    checkpoint = torch.load(ckpt_path)
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)

    model.load_state_dict(model_weights)
    model.eval()

    data_loader = datamodule.test_dataloader()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for _, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > cfg["threshold"]).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    save_result(cfg["exp_name"], rles, filename_and_class)


if __name__ == "__main__":
    main()
