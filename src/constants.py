from datetime import datetime as dt
import logging
import json
import os
from pathlib import Path
from PIL import ImageFile
import torch

# Make sure this gets loaded on any run
ImageFile.LOAD_TRUNCATED_IMAGES = True
START_TIME = dt.now().strftime("%Y_%m_%d_%H_%M")
#############################
#        PATH BASES         #
#############################
SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = Path(os.path.dirname(SRC_DIR))
DATA_DIR = PROJECT_DIR / "data/"
MODEL_DIR = DATA_DIR / "models/"

#############################
#          LOGGING          #
#############################
KLOGGER = logging.getLogger("kodama")
KLOGGER.setLevel(logging.DEBUG)
console_logger = logging.StreamHandler()
file_logger = logging.FileHandler(f"{str(DATA_DIR)}/{START_TIME}_log.log")
# add the handlers to the logger
KLOGGER.addHandler(file_logger)
KLOGGER.addHandler(console_logger)


#############################
#         COMMON FN         #
#############################


def write_json_log(json_dict, save_path=MODEL_DIR, extra_info=f""):
    """
    Function to append a log to a jsonl file
    """
    save_path = str(Path(save_path))
    filename = f"{save_path}/{START_TIME}_{extra_info}.json"
    with open(filename, "a+") as f:
        f.write(json.dumps(json_dict, default=vars) + "\n")


def get_torch_device():
    """
    This function to get device once on import
    """
    if torch.cuda.is_available():
        if torch.cuda.is_tf32_supported():
            torch.set_float32_matmul_precision = "medium"
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def check_cuda(device):
    """
    Checks if cuda is supported and sets some basic flags
    """
    if torch.cuda.is_available() and "cuda" in device:
        return True
    return False


def get_subdir_list(dirs, subdirs=None):
    """
    Takes in list of dirs, and returns list of subdirs with common keys
    """
    if isinstance(subdirs, (str, Path)):
        subdirs = [str(subdirs)]
    if isinstance(dirs, (str, Path)):
        dirs = [str(dirs)]
    res = []
    for this_dir in dirs:
        dir = str(this_dir)
        for sub in os.listdir(dir):
            if subdirs and sub not in subdirs:
                continue
            sub_path = f"{dir}/{sub}"
            if os.path.isdir(sub_path) and sub_path not in res:
                res.append(sub_path)
    return res


#############################
#         CONSTANTS         #
#############################
# Local default torch device
TORCH_DEVICE = get_torch_device()
TORCH_DTYPE = torch.get_autocast_dtype(TORCH_DEVICE)
# Local timezone
LOCAL_TZ = dt.now().astimezone().tzinfo
#############################
#         IMG DIRS          #
#############################
IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
]
IMG_DIR = DATA_DIR / "image/"
# GenImage Dataset
GENIMAGE_DIR = IMG_DIR / "genimage/"
GENIMAGE_DIR = IMG_DIR / "genimage/"
GENIMAGE_DIR = IMG_DIR / "genimage/"
GENIMAGE_DIR = IMG_DIR / "genimage/"
# List of models
GENIMAGE_MODELS = [
    "glide",
    "midjourney",
    "sdv5",
    "vqdm",
    "big_gan",
    "sdv4",
    "wukong",
    "adm",
]
#############################
#         TXT DIRS          #
#############################
TXT_DIR = DATA_DIR / "text/"
