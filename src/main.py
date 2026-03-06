from copy import deepcopy
from datetime import datetime as dt
import gc
import os
import numpy as np
from pathlib import Path
import time
import traceback

import torch
import torch.nn as nn


import constants as cs
from constants import KLOGGER
from utils import find_model_batch_size, find_model_size, benchmark_model_cudnn
from img import etl as etl
from img.model_pretrained import SSPTimmModel
from img.train_img import train_img, test_img, auto_learning_rate

LOCAL_TZ = dt.now().astimezone().tzinfo


ROOT_PATH = Path("/mnt/sn850x/datasets/kodama/image/")
AIORNOT_PATH = ROOT_PATH / "aiornot/"
ARIA_PATH = None  # ROOT_PATH / "aria_dataset/"
ARTIFACT_PATH = ROOT_PATH / "artifact_dataset/"
CNN_DETECT_PATH = ROOT_PATH / "cnn_detection/"
CNN_SYNTH_PATH = ROOT_PATH / "cnn_synth_test/"
FAKEIMAGE_PATH = ROOT_PATH / "FakeImageDataset/"
GENIMAGE_PATH = ROOT_PATH / "genimage/"
HPBENCH_PATH = ROOT_PATH / "hpbench/"
OTHER_PATH = ROOT_PATH / "other/"


def make_model(model_name):
    return SSPTimmModel(
        model_name,
        pre_layers=[],
        post_layers=[
            # nn.Flatten()
            # nn.LazyBatchNorm1d(),
            # nn.ReLu(),
            # nn.LazyLinear(1024),
            # nn.Linear(1024, 256),
            # nn.Linear(256, 1),
        ],
        interpolate_mode="bicubic",
        freeze_base_model=False,
    )


MODEL_NAMES = [
    "eva_large_patch_196",  # use_naflex=True
    "tinyvit_224",
    "tinyvit_s_224",
    "tinyvit_512",
    "coatnet_384",
    "caformer_s_384",
    "caformer_b_384",
]
# Run settings
MAX_EPOCHS = 300
MAX_TRAIN_TIME = 12 * 60 * 60
LOG_INTERVAL = 20 * 60

max_train_files = 700_000_000
# TODO: Bug on subsetting when excluded genimage models
max_test_files = 10_000
max_val_files = 100_000


# OPTIMIZER SETTINGS
# should scale itself with batch size by batch_size. See:
#  https://arxiv.org/abs/1706.02677
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 5e-3
MAX_LEARNING_RATE = 5e-3
BASE_LEARNING_RATE = 1e-3
BASE_BATCH_SIZE = 256
LOSS_FN = nn.BCEWithLogitsLoss()
MAX_GRADIENT = 5.0
MAX_LOSS = 10.0
MIN_LOSS = 0.0


# Hardware Settings
DEVICE = cs.TORCH_DEVICE
BENCH_CUDNN = True
DTYPE = torch.bfloat16
VRAM_SAFETY_BUFFER = 0.9

# Batch sizing settings
BATCH_SIZE = None
NUM_WORKERS = None
IMG_PER_THREAD = 16
BATCH_PREFETCH = 2
TRANSFORM_DEVICE = "cpu"
MAX_BATCH_SIZE = 2048
MAX_CPU_THREADS = 12
COMPILE_TRANSFORMS = False

###### IMG LOADER PARAMS #EXCLUDED_TRAIN_DIRS = []
DATALOADER_OPTIONS = {
    "num_workers": 4,
    "shuffle": True,
    "pin_memory": True,
    "prefetch_factor": BATCH_PREFETCH,
}
PROCESSOR_OPTIONS = {
    "jpg_prob": 0.01,
    "crop_prob": 0.01,
    "blur_prob": 0.01,
    "hflip_prob": 0.05,
    "vflip_prob": 0.01,
    "blur_kernel_size": 3,
    "blur_sigma": (0.1, 2.0),
    "jpg_min_quality": 90,
}
interpolate_mode = "bicubic"
fs = os
random_seed = 42


if __name__ == "__main__":
    KLOGGER.info(DEVICE)
    # This should avoid reloading CUDA in subprocesses:
    if COMPILE_TRANSFORMS and "cuda" in cs.TORCH_DEVICE:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        # https://github.com/pytorch/pytorch/issues/40403
        torch.multiprocessing.set_start_method("spawn")
    for MODEL_NAME in MODEL_NAMES:
        try:
            KLOGGER.info(
                f"\n\n------------\n{MODEL_NAME}\n{PROCESSOR_OPTIONS}\n-----------"
            )
            model = make_model(MODEL_NAME)
            model.to(DEVICE, non_blocking=True)
            model_input_shape = (3, model.crop_size, model.crop_size)
            # Dummy forward call to init model
            model.forward(torch.rand(*(4, *model_input_shape), device=DEVICE))
            model_size_mb = find_model_size(model)
            KLOGGER.info(model)
            KLOGGER.info(
                f"Using: {MODEL_NAME} | {model.crop_size} | {model_size_mb:.3f}MB"
            )
            #################################
            #                               #
            #     Optimize Training Run     #
            #       (make fn from this)     #
            #                               #
            #################################
            # Compile Model
            compile_t = time.time()
            KLOGGER.info(f"Compile t={time.time() - compile_t:.2f}")
            model.compile()
            # Infer batch size
            if not BATCH_SIZE:
                batch_t = time.time()
                KLOGGER.info("Getting automatic batch size...")
                BATCH_SIZE = find_model_batch_size(
                    model,
                    device=DEVICE,
                    dtype=DTYPE,
                    input_shape=model_input_shape,
                    output_shape=1,
                    buffer_size=VRAM_SAFETY_BUFFER,
                )
                gc.collect()
                BATCH_SIZE = min(BATCH_SIZE, MAX_BATCH_SIZE)
                KLOGGER.info(f"{BATCH_SIZE}, t={time.time() - batch_t:.2f}")
            #### Prep batch size dependent values
            if BENCH_CUDNN:
                benchmark_model_cudnn(
                    model,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                    input_shape=model_input_shape,
                    output_shape=1,
                    dtype=DTYPE,
                )
            if not LEARNING_RATE:
                LEARNING_RATE = auto_learning_rate(
                    BATCH_SIZE,
                    base_lr=BASE_LEARNING_RATE,
                    adjustment_factor=BASE_BATCH_SIZE,
                    max_learning_rate=MAX_LEARNING_RATE,
                )
            KLOGGER.info(f"Learning Rate: {LEARNING_RATE}")
            ### NOW PREP LOADERS
            if not NUM_WORKERS:
                NUM_WORKERS = int(
                    min(MAX_CPU_THREADS, BATCH_SIZE / IMG_PER_THREAD)
                )
            train_processor = etl.img_transforms(
                model.crop_size,
                train_mode=True,
                compile=COMPILE_TRANSFORMS,
                dtype=DTYPE,
                **PROCESSOR_OPTIONS,
            )
            img_processor = etl.img_transforms(
                model.crop_size,
                train_mode=False,
                compile=COMPILE_TRANSFORMS,
                dtype=DTYPE,
                **PROCESSOR_OPTIONS,
            )
            if COMPILE_TRANSFORMS:
                train_processor.to(TRANSFORM_DEVICE, non_blocking=True)
                img_processor.to(TRANSFORM_DEVICE, non_blocking=True)
            if "cpu" not in TRANSFORM_DEVICE:
                NUM_WORKERS = 1
            DATALOADER_OPTIONS["batch_size"] = BATCH_SIZE
            DATALOADER_OPTIONS["num_workers"] = NUM_WORKERS
            KLOGGER.info(
                f"Using batch_size: {BATCH_SIZE}, cpu threads: {NUM_WORKERS}"
            )
            # Test Dataloader can have fewer workers
            test_dataloader_opts = deepcopy(DATALOADER_OPTIONS)
            test_dataloader_opts["prefetch_factor"] = 1
            # TODO here: Scale workers on test set to not waste RAM
            # test_dataloader_opts["num_workers"] = int(np.ceil(test_dataloader_opts["num_workers"] // 3))
            test_dataloader_opts["shuffle"] = False

            train_dataloader = etl.train_dataloader(
                max_files=max_train_files,
                crop_size=model.crop_size,
                img_processor=train_processor,
                dataloader_options=DATALOADER_OPTIONS,
                allowed_extensions=cs.IMAGE_EXTENSIONS,
                fs=fs,
                random_seed=random_seed,
                aiornot_path=AIORNOT_PATH,
                aria_path=ARIA_PATH,
                cnn_detect_path=CNN_DETECT_PATH,
                fake_image_path=FAKEIMAGE_PATH,
                genimage_path=GENIMAGE_PATH,
                other_path=None,
            )
            test_dataloader = etl.test_dataloader(
                max_files=max_test_files,
                crop_size=model.crop_size,
                img_processor=img_processor,
                dataloader_options=test_dataloader_opts,
                fs=fs,
                random_seed=random_seed,
                aiornot_path=AIORNOT_PATH,
                aria_path=ARIA_PATH,
                cnn_detect_path=CNN_DETECT_PATH,
                cnn_synth_path=CNN_SYNTH_PATH,
                fake_image_path=FAKEIMAGE_PATH,
                genimage_path=GENIMAGE_PATH,
                hpbench_path=HPBENCH_PATH,
                other_path=None,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )
            gc.collect()
            train_img(
                model,
                optimizer,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                device=DEVICE,
                dtype=DTYPE,
                max_epochs=MAX_EPOCHS,
                loss_fn=LOSS_FN,
                log_interval=LOG_INTERVAL,
                epoch_save_interval=1,
                max_train_time=MAX_TRAIN_TIME,
                max_gradient_norm=MAX_GRADIENT,
                max_loss=MAX_LOSS,
                min_loss=MIN_LOSS,
            )
            # Cleanup RAM for model validation
            train_dataloader = None
            test_dataloader = None
            gc.collect()
            val_dataloader_opts = deepcopy(DATALOADER_OPTIONS)
            val_dataloader_opts["shuffle"] = False
            val_dataloader = etl.test_dataloader(
                max_files=max_val_files,
                crop_size=model.crop_size,
                img_processor=img_processor,
                dataloader_options=val_dataloader_opts,
                fs=fs,
                random_seed=random_seed,
                aiornot_path=None,
                aria_path=None,
                cnn_detect_path=None,
                cnn_synth_path=CNN_SYNTH_PATH,
                fake_image_path=FAKEIMAGE_PATH,
                genimage_path=None,
                hpbench_path=HPBENCH_PATH,
                other_path=None,
            )
            val = test_img(model, val_dataloader, device=DEVICE, dtype=DTYPE)
            for k in val:
                cs.KLOGGER.info(f"{k}: {val[k]:.2f}")
        except Exception as e:
            KLOGGER.error(
                f"\n\n---\nstr(e)\n---\n{traceback.format_exc()}\n---\n\n"
            )
        finally:
            model = None
            torch.cuda.empty_cache()
            gc.collect()
