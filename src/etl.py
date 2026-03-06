import io
import os
from numpy import random as rd
from pathlib import Path
from PIL import Image
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import decode_image, ImageReadMode


from utils import file_list_subset, cast_to_list
import constants as cs
from constants import GENIMAGE_MODELS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The torch transforms loader only takes these
TORCH_IMG_FORMATS = ("png", "jpg", "jpeg", "webp", "gif")

# AIorNot dataset
# https://huggingface.co/datasets/competitions/aiornot
AIORNOT_LABELS = {"real": "label_0", "ai": "label_1"}

# Aria dataset
# https://github.com/AdvAIArtProject/AdvAIArt
ARIA_LABELS = {"real": "real", "ai": "fake"}


# HP Bench is from CNN Detection Dataset
HPBENCH_CATEGORIES = [
    "record",
    "humans",
    "animal",
    "flower",
    "landscape",
    "man",
    "object",
    "woman",
]
HPBENCH_LABELS = {"real": "real", "ai": "fake"}

# CNN Detection Dataset
CNNDETECT_CATEGORIES = [
    "airplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
CNNDETECT_LABELS = {"real": "0_real", "ai": "1_fake"}

# CNN Synth test set from CNN Detect Paper
CNNSYNTH_CATEGORIES = [
    "biggan",
    "crn",
    "cyclegan",
    "deepfake",
    "gaugan",
    "imle",
    "progan",
    "san",
    "seeingdark",
    "stargan",
    "stylegan",
    "stylegan2",
    "whichfaceisreal",
]
CNNSYNTH_LABELS = {"real": "0_real", "ai": "1_fake"}

# Genimage Dataset
GENIMAGE_CATEGORIES = [
    "vqdm",
    "big_gan",
    "glide",
    "adm",
    "sdv5",
    "midjourney",
    "wukong",
    "sdv4",
]
GENIMAGE_LABELS = {"real": "0_real", "ai": "1_fake"}


def webp_compress(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality)
    return Image.open(buffer)


def img_transforms(
    crop_size=224,
    dtype=cs.TORCH_DTYPE,
    compile=False,
    train_mode=True,
    crop_prob=0.3,
    blur_prob=0.1,
    blur_kernel_size=3,
    blur_sigma=(0.1, 2.0),
    jpg_prob=0.1,
    jpg_min_quality=90,
    hflip_prob=0.1,
    vflip_prob=0.1,
    norm_mean=IMAGENET_MEAN,
    norm_std=IMAGENET_STD,
    interpolation=v2.InterpolationMode.BICUBIC,
):
    """
    Creates default transforms for training

    NOTES:
        Details matter in the image input
        AI Generated Images tend to be created whole
        They are uncommonly cropped
        They are often JPEG compressed online
    """
    tx = []
    # PIL is HxWxC, tensors are CxHxW
    # F.pil_to_tensor(pic) performs a deep copy
    # Resizing prefers channel-last (PIL)
    # Use tensors instead of PIL images
    # Use torch.uint8 dtype, especially for resizing
    # v2.query_chw --> Gets channel, height, width
    tx.append(v2.ToImage())
    tx.append(v2.RGB())  # F.grayscale_to_rgb
    # uint8 is much faster for transforming after resizing
    tx.append(v2.ToDtype(torch.uint8, scale=True))  # F.to_dtype
    if train_mode:
        # F.pad as needed at the end
        # F.jpeg
        # F.vertical_flip
        # F.horizontal_flip
        # SHOULD BE RARELY USED
        #   F.gaussian_blur
        #   F.gaussian_noise
        #   F.adjust_sharpness : 0->1 blurs >1 sharpens
        #   F.rotate | F.perspective or just F.affine --> + with center crops
        #   F.autocontrast max pixel -> white, min pixel -> black
        #   F.posterize (randomly reduces color bits)
        if hflip_prob:
            tx.append(v2.RandomHorizontalFlip(p=hflip_prob))
        if vflip_prob:
            tx.append(v2.RandomVerticalFlip(p=vflip_prob))
        if jpg_prob:
            tx.append(
                v2.RandomApply(
                    torch.nn.ModuleList([v2.JPEG((jpg_min_quality, 100))]), p=jpg_prob
                )
            )
        if blur_prob:
            tx.append(
                v2.RandomApply(
                    torch.nn.ModuleList(
                        [v2.GaussianBlur(blur_kernel_size, blur_sigma)]
                    ),
                    p=blur_prob,
                )
            )
        if crop_prob:
            tx.append(
                v2.RandomApply(
                    torch.nn.ModuleList(
                        [v2.RandomCrop((crop_size, crop_size), pad_if_needed=True)]
                    ),
                    p=crop_prob,
                )
            )
    tx.append(v2.Resize((crop_size, crop_size), interpolation=interpolation))
    # Cast to tensor now
    tx.append(v2.ToDtype(dtype, scale=True))
    if train_mode and norm_mean and norm_std:
        tx.append(v2.Normalize(mean=norm_mean, std=norm_std))
    if compile:
        tx = torch.nn.Sequential(*tx)
        tx.compile()
        return tx
    else:
        return v2.Compose(tx)


class KodamaImageDataset(Dataset):
    def __init__(
        self,
        input_path,
        img_processor,
        crop_size=256,
        real_labels=None,
        ai_labels=None,
        max_files=None,
        included_paths=None,
        allowed_extensions=cs.IMAGE_EXTENSIONS,
        excluded_paths=None,
        fs=os,
        random_seed=None,
        shuffle=True,
        skip_errors=True,
        verbose=True,
    ):
        super().__init__()
        self.img_processor = img_processor
        self.crop_size = crop_size
        self.skip_errors = skip_errors
        self.bad_image_idx = []
        self.verbose = verbose
        if max_files:
            max_files = max_files // 2
        # Get real images (exclude AI ones)
        excluded_paths = cast_to_list(excluded_paths)
        real_labels = cast_to_list(real_labels)
        ai_labels = cast_to_list(ai_labels)
        real_excluded_paths = ai_labels + excluded_paths
        ai_excluded_paths = real_labels + excluded_paths
        start_t = time.time()
        cs.KLOGGER.debug(f"\n---\nloading: {input_path}")
        cs.KLOGGER.debug(f"load real img:")
        self.real_images = file_list_subset(
            input_path=input_path,
            max_files=max_files,
            included_paths=included_paths,
            excluded_paths=real_excluded_paths,
            fs=fs,
            random_seed=random_seed,
        )
        cs.KLOGGER.debug(
            f"t={time.time() - start_t:.2f}, images:{len(self.real_images)}"
        )
        # Get AI images
        start_t = time.time()
        cs.KLOGGER.debug(f"load ai img:")
        self.ai_images = file_list_subset(
            input_path=input_path,
            max_files=max_files,
            included_paths=included_paths,
            excluded_paths=ai_excluded_paths,
            allowed_extensions=allowed_extensions,
            fs=fs,
            random_seed=random_seed,
        )
        cs.KLOGGER.debug(f"t={time.time() - start_t:.2f}, img:{len(self.ai_images)}")
        if shuffle:
            shuffle_t = time.time()
            if self.verbose:
                cs.KLOGGER.debug("shuffling datasets...")
            random.shuffle(self.real_images)
            random.shuffle(self.ai_images)
            if self.verbose:
                cs.KLOGGER.debug(f"t={time.time() - shuffle_t:.2f}")
        self.images = self.real_images + self.ai_images
        self.n_real = len(self.real_images)
        self.n_fake = len(self.ai_images)
        if self.n_real == 0:
            raise ValueError("No images with real label!")
        if self.n_fake == 0:
            raise ValueError("No images with AI label!")
        self.n_images = len(self.images)
        self.labels = torch.cat(
            (
                torch.ones(self.n_real),
                torch.zeros(self.n_fake),
            )
        )

    def __getitem__(self, index):
        try:
            if index in self.bad_image_idx:
                index = rd.randint(0, self.n_images)
            img_path = self.images[index]
            label = self.labels[index]
            try:
                img = decode_image(img_path, mode=ImageReadMode.RGB)
                image = self.img_processor(img)
            except RuntimeError as re:
                if self.verbose:
                    cs.KLOGGER.warning(f"Decode err: {img_path} | {re}")
                with open(img_path, "rb") as f:
                    img = Image.open(f).convert("RGB")
                image = self.img_processor(img)
            return image, label
        except Exception as e:
            if self.verbose:
                cs.KLOGGER.error(f"Bad image:{self.images[index]} | {str(e)}")
            if not self.skip_errors:
                raise e
            self.bad_image_idx.append(index)
            index = rd.randint(0, self.n_images)
            return self.__getitem__(index)

    def __len__(self):
        return self.n_images


###############################
#                             #
#         DATASETS            #
#                             #
###############################


def aiornot_dataset(
    input_path,
    img_processor,
    key="train",
    real_label=AIORNOT_LABELS["real"],
    ai_label=AIORNOT_LABELS["ai"],
    *args,
    **kwargs,
):
    assert key in ["train", "test"]
    input_path = Path(input_path)
    excl_key = "test" if key == "train" else "train"
    kwargs["excluded_paths"] = [excl_key]
    return KodamaImageDataset(
        input_path=input_path,
        img_processor=img_processor,
        real_labels=real_label,
        ai_labels=ai_label,
        *args,
        **kwargs,
    )


def aria_dataset(
    input_path,
    img_processor,
    real_label=ARIA_LABELS["real"],
    ai_label=ARIA_LABELS["ai"],
    *args,
    **kwargs,
):
    input_path = Path(input_path)
    return KodamaImageDataset(
        input_path=str(input_path),
        img_processor=img_processor,
        real_labels=real_label,
        ai_labels=ai_label,
        *args,
        **kwargs,
    )


def cnn_detect_dataset(
    input_path,
    img_processor,
    key="train",  # train / test / val
    real_label=CNNDETECT_LABELS["real"],
    ai_label=CNNDETECT_LABELS["ai"],
    included_paths=CNNDETECT_CATEGORIES,
    *args,
    **kwargs,
):
    assert key in ["train", "test", "val"]
    input_path = Path(input_path)
    return KodamaImageDataset(
        input_path=str(input_path / key),
        img_processor=img_processor,
        real_labels=real_label,
        ai_labels=ai_label,
        included_paths=included_paths,
        *args,
        **kwargs,
    )


def cnn_synth_dataset(
    input_path,
    img_processor,
    real_label=CNNSYNTH_LABELS["real"],
    ai_label=CNNSYNTH_LABELS["ai"],
    included_paths=CNNSYNTH_CATEGORIES,
    *args,
    **kwargs,
):
    return KodamaImageDataset(
        input_path=input_path,
        img_processor=img_processor,
        real_labels=real_label,
        ai_labels=ai_label,
        included_paths=included_paths,
        *args,
        **kwargs,
    )


def genimage_dataset(
    input_path,
    img_processor,
    key="train",
    real_label=GENIMAGE_LABELS["real"],
    ai_label=GENIMAGE_LABELS["ai"],
    included_paths=GENIMAGE_CATEGORIES,
    *args,
    **kwargs,
):
    assert key in ["train", "test"]
    excl_key = "val" if key == "train" else "train"
    kwargs["excluded_paths"] = [excl_key]
    return KodamaImageDataset(
        input_path=input_path,
        img_processor=img_processor,
        real_labels=real_label,
        ai_labels=ai_label,
        included_paths=included_paths,
        *args,
        **kwargs,
    )


def hpbench_dataset(
    input_path,
    img_processor,
    real_label=HPBENCH_LABELS["real"],
    ai_label=HPBENCH_LABELS["ai"],
    included_paths=HPBENCH_CATEGORIES,
    *args,
    **kwargs,
):
    return KodamaImageDataset(
        input_path=input_path,
        img_processor=img_processor,
        real_labels=real_label,
        ai_labels=ai_label,
        included_paths=included_paths,
        *args,
        **kwargs,
    )


def train_dataloader(
    aiornot_path=None,
    aria_path=None,
    cnn_detect_path=None,
    fake_image_path=None,
    genimage_path=None,
    other_path=None,
    crop_size=256,
    img_processor=img_transforms,
    dataloader_options={
        "batch_size": 32,
        "num_workers": 4,
        "shuffle": True,
        "pin_memory": True,
    },
    max_files=None,
    allowed_extensions=cs.IMAGE_EXTENSIONS,
    excluded_paths=None,
    fs=os,
    random_seed=None,
    skip_errors=True,
    verbose=True,
):
    start_t = time.time()
    opts = {
        "max_files": max_files,
        "allowed_extensions": allowed_extensions,
        "excluded_paths": excluded_paths,
        "fs": fs,
        "random_seed": random_seed,
        "skip_errors": skip_errors,
        "verbose": verbose,
    }
    datasets = []
    if aiornot_path:
        datasets.append(
            aiornot_dataset(aiornot_path, img_processor, key="train", **opts)
        )
    if aria_path:
        datasets.append(aria_dataset(aria_path, img_processor, **opts))
    if genimage_path:
        datasets.append(
            genimage_dataset(genimage_path, img_processor, key="train", **opts)
        )
    if cnn_detect_path:
        datasets.append(
            cnn_detect_dataset(cnn_detect_path, img_processor, key="train", **opts)
        )
    datasets = torch.utils.data.ConcatDataset(datasets)
    dloader = DataLoader(datasets, **dataloader_options)
    return dloader


def test_dataloader(
    aiornot_path=None,
    aria_path=None,
    cnn_synth_path=None,
    cnn_detect_path=None,
    fake_image_path=None,
    genimage_path=None,
    hpbench_path=None,
    other_path=None,
    crop_size=256,
    img_processor=img_transforms,
    dataloader_options={
        "batch_size": 32,
        "num_workers": 4,
        "shuffle": True,
        "pin_memory": True,
    },
    models=GENIMAGE_MODELS,
    max_files=None,
    allowed_extensions=cs.IMAGE_EXTENSIONS,
    excluded_paths=None,
    fs=os,
    random_seed=None,
    skip_errors=True,
    verbose=True,
):
    start_t = time.time()
    opts = {
        "max_files": max_files,
        "allowed_extensions": allowed_extensions,
        "excluded_paths": excluded_paths,
        "fs": fs,
        "random_seed": random_seed,
        "skip_errors": skip_errors,
        "verbose": verbose,
    }
    datasets = []
    if aiornot_path:
        datasets.append(
            aiornot_dataset(aiornot_path, img_processor, key="test", **opts)
        )
    if cnn_detect_path:
        datasets.append(
            cnn_detect_dataset(cnn_detect_path, img_processor, key="test", **opts)
        )
        datasets.append(
            cnn_detect_dataset(cnn_detect_path, img_processor, key="val", **opts)
        )
    if cnn_synth_path:
        datasets.append(cnn_synth_dataset(cnn_detect_path, img_processor, **opts))
    if hpbench_path:
        datasets.append(hpbench_dataset(hpbench_path, img_processor, **opts))
    if genimage_path:
        datasets.append(
            genimage_dataset(genimage_path, img_processor, key="test", **opts)
        )
    datasets = torch.utils.data.ConcatDataset(datasets)
    dloader = DataLoader(datasets, **dataloader_options)
    return dloader


# Validation: HP Bench, cnn_synth, aiornot (test), cnn_detect (test)
# TODO: Make train/test split sentry/ARIA/real scrapes dataset
# TODO: Make eval function per dataset to get scores for each
