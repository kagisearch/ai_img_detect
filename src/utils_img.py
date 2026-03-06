import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def plot_img(imgs, **imshow_kwargs):
    if not isinstance(imgs, list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]
    num_cols = len(imgs)
    _, axs = plt.subplots(nrows=1, ncols=num_cols, squeeze=False)
    for col_idx, img in enumerate(imgs):
        img = F.to_image(img)
        if img.dtype.is_floating_point and img.min() < 0:
            # Poor man's re-normalization for the colors to be OK-ish. This
            # is useful for images coming out of Normalize()
            img -= img.min()
            img /= img.max()

        img = F.to_dtype(img, torch.uint8, scale=True)
        ax = axs[0, col_idx]
        ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
