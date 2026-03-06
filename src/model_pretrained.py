import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np

# Use NAFLEX by default:
#   https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eva.py#L1272
import os

os.environ["TIMM_USE_NAFLEX"] = "1"


import timm


# https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet-a.csv
# speed | top1 imagenet_a | param_count
TIMM_MODELS = {
    # 1100FPS | 2 | 2.0m
    "tinynet_106": {
        "model_key": "tinynet_e.in1k",
        "crop_size": 106,
    },
    # xxxxFPS | 3 | 2.3m
    "tinynet_152": {
        "model_key": "tinynet_d.in1k",
        "crop_size": 152,
    },
    # Useless -- doesnt work on bf16 on large batches
    # Trains on low batch, LR 1e-4
    # 2200FPS | 3.7 | 3.8m | 15MB
    "mobilenet_v4_s": {
        "model_key": "mobilenetv4_conv_small.e2400_r224_in1k",
        "crop_size": 224,
    },
    # xxxFPS | 3.2 | 2.4m
    "tinynet_184": {
        "model_key": "tinynet_c.in1k",
        "crop_size": 184,
    },
    # xxxFPS | 4.3 | 3.7m
    "tinynet_188": {
        "model_key": "tinynet_b.in1k",
        "crop_size": 188,
    },
    # BF16 works on smaller batch sizes
    # no float16 though gives error on pow() compiling
    # xxxFPS | 5.3 | 8m | 33MB
    "efficientnet_192": {
        "model_key": "tf_efficientnetv2_b1.in1k",
        "crop_size": 192,
    },
    # xxxFPS | 6.7 | 6.2m
    "tinynet_192": {
        "model_key": "tinynet_a.in1k",
        "crop_size": 192,
    },
    # 1400FPS | 16 | 11m |
    "mobilenet_v4": {
        "model_key": "mobilenetv4_hybrid_medium.e500_r224_in1k",
        "crop_size": 224,
    },
    # 510FPS | 11.7 | 25.6m
    "ecaresnet_160": {
        "model_key": "ecaresnet50t.a3_in1k",
        "crop_size": 160,
    },
    # xxFPS | 33 | 11m
    "tinyvit_s_224": {
        "model_key": "tiny_vit_11m_224.dist_in22k_ft_in1k",
        "crop_size": 224,
    },
    # 75FPS | 15 | 5.59m
    "edgenext": {
        "model_key": "edgenext_small.usi_in1k",
        "crop_size": 256,
    },
    # 60FPS | 26 | 74m
    "coatnet_nano": {
        "model_key": "coatnet_rmlp_nano_rw_224.sw_in1k",
        "crop_size": 224,
    },
    # 500FPS | 44 | 21m
    "tinyvit_224": {
        "model_key": "tiny_vit_21m_224.dist_in22k_ft_in1k",
        "crop_size": 224,
    },
    # 55FPS | 36.4 | 10m
    "fastvit_apple": {
        "model_key": "fastvit_sa24.apple_dist_in1k",
        "crop_size": 256,
    },
    #### Venerable Resnet50
    # 28FPS | 11 | 25m | 90MB
    "resnet50": {
        "model_key": "resnet50d.ra4_e3600_r224_in1k",
        "crop_size": 224,
    },
    # Doesnt work on BF16 at all
    # 25FPS | 45 | 3.8m | 15MB
    "mobilenet_v4_s_448": {
        "model_key": "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k",
        "crop_size": 448,
    },
    # 16FPS | 42 | 32m
    "nextvit_small_384": {
        "model_key": "nextvit_small.bd_ssld_6m_in1k_384",
        "crop_size": 384,
        "interpolation": "bicubic",
    },
    ##### Low FPS Models #####
    # 93FPS | 58 | 22m
    "tinyvit_512": {
        "model_key": "tiny_vit_21m_512.dist_in22k_ft_in1k",
        "crop_size": 512,
        "interpolation": "bicubic",
    },
    # 80FPS | 72 | 74m
    "coatnet_384": {
        "model_key": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        "crop_size": 384,
    },
    # use_naflex=True
    # 116FPS | 75 | 304m
    "eva_large_patch_196": {
        "model_key": "eva_large_patch14_196.in22k_ft_in22k_in1k",
        "crop_size": 196,
    },
    # xxFPS | 71 | 39m
    "caformer_s_384": {
        "model_key": "caformer_s36.sail_in22k_ft_in1k_384",
        "crop_size": 384,
    },
    # 3.2FPS | 67 | 66m
    "efficientnet_600": {
        "model_key": "tf_efficientnet_b7.ns_jft_in1k",
        "crop_size": 600,
    },
    # 3.0FPS | 80 | 99m
    "caformer_b_384": {
        "model_key": "caformer_b36.sail_in22k_ft_in1k_384",
        "crop_size": 384,
    },
    # 3.0FPS | 78 | 87m
    "eva02_b": {
        "model_key": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        "crop_size": 448,
    },
    # 1.6FPS | 81 | 120M
    "maxvit_b_512": {
        "model_key": "maxvit_base_tf_512.in21k_ft_in1k",
        "crop_size": 512,
    },
}


class SSPTimmModel(nn.Module):
    def __init__(
        self,
        base_model_key,
        pre_layers=[],
        post_layers=[
            # nn.LazyLinear(1),
        ],
        interpolate_mode="bicubic",
        freeze_base_model=False,
        clamp_min=-1e9,
        clamp_max=1e9,
    ):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.pre_layers = None
        self.post_layers = None
        if pre_layers:
            self.pre_layers = nn.ModuleList(pre_layers)
        if post_layers:
            self.post_layers = nn.ModuleList(post_layers)
        num_classes = 1
        if post_layers:
            num_classes = 0
        crop_size = TIMM_MODELS[base_model_key]["crop_size"]
        self.model_name = base_model_key
        self.base_model = timm.create_model(
            TIMM_MODELS[self.model_name]["model_key"],
            pretrained=True,
            num_classes=num_classes,
        )
        self.interpolate_mode = interpolate_mode
        self.crop_size = int(crop_size)
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # This interpolation is not needed
        # x = F.interpolate(x, (self.crop_size, self.crop_size), mode=self.interpolate_mode)
        if self.pre_layers:
            for l in self.pre_layers:
                x = l(x)
        x = self.base_model(x)
        if self.post_layers:
            for l in self.post_layers:
                x = l(x)

        return x
