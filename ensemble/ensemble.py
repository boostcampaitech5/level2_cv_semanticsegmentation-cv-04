# python native
import os
import json
import random

# from argparse import ArgumentParser

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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
NUM_CLASS = 29


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


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


def ensemble(CSV_FOLDER_PATH):
    output_list = os.listdir(CSV_FOLDER_PATH)
    output_list.sort(reverse=True)

    df_list = []

    for output in output_list:
        df_list.append(pd.read_csv(f"{CSV_FOLDER_PATH}/{output}"))

    # submission dataframe
    submission = pd.DataFrame()
    submission["image_name"] = df_list[0]["image_name"]
    submission["class"] = df_list[0]["class"]

    num_models = len(df_list)
    NUM_PICTURES = 300  # test 이미지 개수
    rles = []
    for image_idx in range(NUM_PICTURES):
        for class_id in range(NUM_CLASS):
            pred = np.zeros((2048, 2048))
            idx = image_idx * NUM_CLASS + class_id
            for df_idx in range(num_models):
                pred += decode_rle_to_mask(
                    df_list[df_idx].iloc[idx]["rle"], height=2048, width=2048
                )
            pred[pred >= (num_models) // 2] = 1
            rles.append(encode_mask_to_rle(pred))
            # submission.iloc[idx]['rle'] = encode_mask_to_rle(pred)

    submission["rle"] = rles
    submission.to_csv("./hard_voted_output.csv", index=False)


if __name__ == "__main__":
    CSV_FOLDER_PATH = "/opt/ml/ensemble/csv_files"
    ensemble(CSV_FOLDER_PATH)
