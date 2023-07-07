import numpy as np
import pandas as pd
from collections import defaultdict
import os
from tqdm.auto import tqdm
import cv2
import torch

from mmengine.config import Config
from mmengine.dataset import Compose

from mmseg.apis import init_model
config_path = '/opt/ml/mmsegmentation/exp/upernet_03/upernet_03.py'
checkpoint_path = '/opt/ml/mmsegmentation/exp/upernet_03/iter_17600.pth'

# csv 파일 이름
CSV_FILE = "/opt/ml/mmsegmentation/exp/upernet_03/iter_17600_best_model.csv"

# 테스트 데이터 경로 입력
IMAGE_ROOT = "/opt/ml/input/data/test/DCM/"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

cfg = Config.fromfile(config_path)
model = init_model(config_path, checkpoint_path, device='cuda:0')

def _prepare_data(imgs, model):
    for t in cfg.test_pipeline:
        if t.get('type') in ['LoadXRayAnnotations', 'TransposeAnnotations']:
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data


pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def test(model):
    rles = []
    filename_and_class = []
    with torch.no_grad():
        for image_names in tqdm(sorted(pngs), total=len(pngs)):
            image_path = os.path.join(IMAGE_ROOT, image_names)
            image = cv2.imread(image_path)

            data = _prepare_data(image, model)
            results = model.test_step(data)
            outputs = results[0].pred_sem_seg.data
            
            outputs = outputs.cpu()
            for c, segm in enumerate(outputs):
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{image_names}")
                    
    return rles, filename_and_class

rles, filename_and_class = test(model)


classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]
df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv(CSV_FILE, index=False)
