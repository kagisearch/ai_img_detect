from datetime import datetime as dt
from itertools import islice
import os
from numpy import random as rd
import pandas as pd
from pathlib import Path
from PIL import Image
import random
import time
import tqdm

import timm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 100)
pd.set_option('future.no_silent_downcasting', True)

# Get relevant directories
if "notebooks" in os.getcwd():
    TOP_DIR = Path(os.path.dirname(os.path.abspath(os.getcwd()))) 
    SRC_DIR = TOP_DIR / "src"
    # Move to project root dir
    os.chdir(SRC_DIR)

import constants as cs
from img import etl as etl
from img.utils_img import plot_img
from utils import subdir_file_list

print(cs.TORCH_DEVICE)


MAX_FILES = 500


ROOT_PATH = Path("/mnt/sn850x/datasets/kodama/image/")
AIORNOT_PATH = ROOT_PATH / "aiornot/"
ARIA_PATH = ROOT_PATH / "aria_dataset/"
ARTIFACT_PATH = ROOT_PATH / "artifact_dataset/"
CNN_DETECT_PATH = ROOT_PATH / "cnn_detection/"
CNN_SYNTH_PATH = ROOT_PATH / "cnn_synth_test/"
FAKEIMAGE_PATH = ROOT_PATH / "FakeImageDataset/"
GENIMAGE_PATH = ROOT_PATH / "genimage/"
HPBENCH_PATH = ROOT_PATH / "hpbench/"
OTHER_PATH = ROOT_PATH / "other/"


dlt = etl.test_dataloader(
    max_files=MAX_FILES,
    aiornot_path=AIORNOT_PATH,
    aria_path=ARIA_PATH,
    cnn_detect_path=CNN_DETECT_PATH,
    cnn_synth_path=CNN_SYNTH_PATH,
    fake_image_path=FAKEIMAGE_PATH,
    genimage_path=GENIMAGE_PATH,
    hpbench_path=HPBENCH_PATH,
    other_path=None,
    crop_size=256,
    img_processor=etl.img_transforms(),
    dataloader_options={
        "batch_size": 4,
        "num_workers": 4,
        "shuffle": True,
        "pin_memory": True,
    },
    allowed_extensions=cs.IMAGE_EXTENSIONS,
    excluded_paths=None,
    fs=os,
    random_seed=None,
    skip_errors=True,
    verbose=True,
)

train = list(islice(dlt, MAX_FILES))
train[0][0].shape


# In[ ]:


img_path = "/mnt/sn850x/datasets/kodama/image/hpbench/fake/flower/11.png"
img = Image.open(img_path)
print(img.mode)

img = decode_image(img_path)#, mode=ImageReadMode.RGB)
print(img.shape, img.dtype)
img = v2.RGB()(img)
print(img.shape, img.dtype)

plot_img(img)


# In[ ]:


img


# In[ ]:


raise ValueError


# # REMAPPING DATASETS

# # AIOrnot

# In[ ]:


import pandas as pd
import shutil

AIORNOT_TRAIN = AIORNOT_PATH / "train"
AIORNOT_TRAIN_LABEL_0 = AIORNOT_TRAIN / "label_0"
AIORNOT_TRAIN_LABEL_1 = AIORNOT_TRAIN / "label_1"

df_t = pd.read_csv(AIORNOT_PATH / "train.csv")
for f in df_t.loc[df_t.label == 1]['id'].values:
    shutil.move(
        str(AIORNOT_TRAIN / f), 
        str(AIORNOT_TRAIN_LABEL_1 / f)
    )


AIORNOT_TEST = AIORNOT_PATH / "test"
AIORNOT_TEST_LABEL_0 = AIORNOT_TEST / "label_0"
AIORNOT_TEST_LABEL_1 = AIORNOT_TEST / "label_1"

df_t = pd.read_csv(AIORNOT_PATH / "solution.csv")
for f in df_t.loc[df_t.label == 1]['id'].values:
    shutil.move(
        str(AIORNOT_TEST / f), 
        str(AIORNOT_TEST_LABEL_1 / f)
    )


# # Artifact

# In[6]:


import pandas as pd
import shutil

ARTIFACT_REAL = ARTIFACT_PATH / "label_real"
ARTIFACT_AI = ARTIFACT_PATH / "label_fake"

for model in os.listdir(ARTIFACT_PATH):
    model_path = ARTIFACT_PATH / model
    model_dir = os.listdir(model_path)
    if "metadata.csv" not in model_dir:
        print(f"\n\n---\nNo Metadata in {model}: {str(model_dir)}\n---\n\n")
        continue
    df = pd.read_csv(model_path / "metadata.csv")
    df_fake = df.loc[df.target > 0]
    df_real = df.loc[df.target > 0]
    print(f"\n\n---\n{model}\n\tFake: {len(df_fake)}\n\tReal: {len(df_real)}\n---\n\n")



    # for f in df_t.loc[df_t.label == 1]['id'].values:
    #     shutil.move(
    #         str(AIORNOT_TRAIN / f), 
    #         str(AIORNOT_TRAIN_LABEL_1 / f)
    #     )


# In[ ]:




