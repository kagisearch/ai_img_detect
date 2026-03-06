import gc
import numpy as np
import os
from pathlib import Path
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable

import constants as cs
from torch.utils.data import Sampler


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def cast_to_list(x):
    if x is None:
        return []
    if isinstance(x, (bool, int, float, str)):
        return [x]
    if isinstance(x, list):
        return x
    else:
        return []


def find_model_size(model):
    """
    Finds size of torch model
    from:
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def find_model_batch_size(
    model,
    device,
    input_shape=(3, 256, 256),
    output_shape=1,
    max_iter=10,
    buffer_size=0.9,
    verbose=True,
    dtype=cs.TORCH_DTYPE,
) -> int:
    if isinstance(output_shape, int):
        output_shape = (output_shape,)
    # Start by jumping with a factor of 8 to find max RAM fast
    scale_factor = 8
    is_cuda = cs.check_cuda(device)
    prev_bench_val = False
    if is_cuda:
        # Dont benchmark while playing around model sizes
        prev_bench_val = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        scale_factor = 16
    model.to(device)
    model.train(True)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.GradScaler(device, enabled=True)
    batch_size = 4
    prev_batch_size = 2
    for _ in range(max_iter):
        try:
            # dummy inputs and targets
            inputs = torch.rand(*(batch_size, *input_shape), device=device, dtype=dtype)
            targets = torch.rand(*(batch_size, *output_shape), device=device, dtype=dtype)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, enabled=is_cuda, cache_enabled=False, dtype=dtype):
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            prev_batch_size = batch_size
            batch_size = int(np.ceil(batch_size * scale_factor))
        except RuntimeError as re:
            if verbose:
                if "memory" not in str(re):
                    print(re)
                print(f"{batch_size} too big ->", end=" ")
            if scale_factor > 2:
                scale_factor = 2
                batch_size = prev_batch_size * scale_factor
            else:
                # New search point is halfway between
                batch_diff = batch_size - prev_batch_size
                binary_search = int(prev_batch_size + batch_diff / 2)
                # We scale down search on succesful retry
                # Since we start aggressively with a size of 8, scale back
                scale_factor = min(2, binary_search / prev_batch_size)
                batch_size = binary_search
            if verbose:
                print(f"{batch_size}")
    final_size = int(np.ceil(prev_batch_size * buffer_size))
    if verbose:
        print(f"{final_size} ({batch_size} * {buffer_size})")
    del model, optimizer
    if is_cuda:
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = prev_bench_val
    torch.clear_autocast_cache()
    gc.collect()
    return final_size


def benchmark_model_cudnn(
    model,
    device,
    batch_size=32,
    input_shape=(3, 256, 256),  # input shape
    output_shape=1,  # num_classes
    verbose=True,
    dtype=None,
):
    """
    Runs single loop of CUDNN Benchmark
    """
    if not cs.check_cuda(device):
        return
    prev_bench_val = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    if isinstance(output_shape, int):
        output_shape = (output_shape,)
    if verbose:
        print("benchmarking cudnn...", end="")
    # Now that we know the size of the network, bench it
    bench_t = time.time()
    inputs = torch.rand(*(batch_size, *input_shape), device=device, dtype=dtype)
    targets = torch.rand(*(batch_size, *output_shape), device=device, dtype=dtype)
    with torch.autocast(device_type=device, cache_enabled=False, dtype=dtype):
        outputs = model(inputs)
        _ = F.mse_loss(targets, outputs)
    if verbose:
        print(f"t={time.time() - bench_t:.2f}")
    torch.cuda.empty_cache()
    torch.clear_autocast_cache()
    torch.backends.cudnn.benchmark = prev_bench_val


def subdir_file_list(
    input_path,
    max_files=None,
    allowed_extensions=None,
    excluded_paths=None,
    fs=os,
    random_seed=None,
):
    """
    Gets list of files from deeply nested subdirectories

    fs: filesystem to use
        Normally "os" (local files)
        can also be a gcsfs instance:
        ```
        gcs = gcsfs.GCSFileSystem()
        subdir_file_list('my-bucket', fs=gcs)
        ```
    """
    if random_seed:
        set_random_seed(random_seed)
    # Annoying fix since .endswith() takes only tuples
    if allowed_extensions:
        if not isinstance(allowed_extensions, list):
            allowed_extensions = [allowed_extensions]
        # lowercase the extensions on both sides
        allowed_extensions = tuple([e.lower() for e in allowed_extensions])
    n_files = 0
    res = []
    for cur_path, dirs, files in fs.walk(input_path):
        if max_files and n_files >= max_files:
            break
        # Modifying dirs prunes next dirs visited by os.walk
        if excluded_paths:
            dirs[:] = [d for d in dirs if d not in excluded_paths]
        # If there's no files, skip the file handling
        if not files:
            continue
        flist = files
        if random_seed:
            sample_size = min(max_files, len(files)) if max_files else len(files)
            flist = random.sample(files, k=sample_size)
        for file in flist:
            if allowed_extensions and not file.lower().endswith(allowed_extensions):
                continue
            full_file_path = str(Path(cur_path) / file)
            res.append(full_file_path)
            n_files += 1
            if max_files and n_files >= max_files:
                break
    return res


def file_list_subset(
    input_path,
    max_files=None,
    allowed_extensions=cs.IMAGE_EXTENSIONS,
    excluded_paths=None,
    included_paths=None,
    fs=os,
    random_seed=None,
):
    """
    Wrapper around utils.subdir_file_list that gets even split of
    max files from models
    """
    if not os.path.isdir(input_path):
        raise ValueError(f"Path {input_path} is not a directory")
    excluded_paths = cast_to_list(excluded_paths)
    included_paths = cast_to_list(included_paths)
    if not included_paths:
        for _, dirs, _ in fs.walk(input_path):
            included_paths = [d for d in dirs if d not in excluded_paths]
            break  # only look at top paths
    if max_files and max_files > len(included_paths):
        max_files = max_files // len(included_paths)
    res = []
    path_files = {}
    for this_path in included_paths:
        this_exclusions = [m for m in included_paths if m != this_path]
        this_files = subdir_file_list(
            input_path,
            max_files=max_files,
            allowed_extensions=allowed_extensions,
            excluded_paths=excluded_paths + this_exclusions,
            fs=fs,
            random_seed=random_seed,
        )
        path_files[this_path] = len(this_files)
        res.extend(this_files)
    cs.KLOGGER.debug(f"{path_files}")
    return res
