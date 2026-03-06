from datetime import datetime as dt
import json
import numpy as np
import os
from pathlib import Path
from sklearn import metrics
import time
import torch
import torch.nn as nn
import traceback

from constants import TORCH_DEVICE, LOCAL_TZ, KLOGGER
import constants as cs


def save_model(model, save_path=cs.MODEL_DIR, extra_info=""):
    save_path = str(Path(save_path))
    now_str = dt.now().strftime("%m_%d_%H")
    torch.save(model.state_dict(), f"{save_path}/{now_str}_{extra_info}.pth")


def auto_learning_rate(batch_size=32, n_gpu=1, base_lr=0.001, adjustment_factor=256, max_learning_rate=0.05):
    """
    Learning rate from batch size from:
        https://arxiv.org/pdf/1706.02677
    (see 5.1)
    """
    return min(max_learning_rate, base_lr * (n_gpu * batch_size) / adjustment_factor)


################
#    TRAIN     #
################
def train_img(
    model,
    optimizer,
    train_dataloader,
    test_dataloader=None,
    device=TORCH_DEVICE,
    max_epochs=1,
    loss_fn=nn.BCEWithLogitsLoss(),
    log_interval=60,
    epoch_save_interval=1,
    save_path=cs.MODEL_DIR,
    val_metrics=[
        "accuracy",
        "precision",
        "recall",
        "correct_guesses",
        "nan_outputs",
        "logits_min",
        "logits_q25",
        "logits_q50",
        "logits_q75",
        "logits_max",
    ],
    dtype=cs.TORCH_DTYPE,
    max_train_time=None,
    max_gradient_norm=5.0,
    max_loss=None,
    min_loss=None,
):
    """
    Training loop for image models
    """
    assert os.path.isdir(save_path)
    save_path = Path(save_path)
    model_name = ""
    # Grad scaling for AMP training
    scaler = torch.GradScaler(device, enabled=True)
    if hasattr(model, "model_name"):
        model_name = model.model_name
    if not max_train_time:
        max_train_time = np.inf
    start_t = time.time()
    last_log_t = 0
    for epoch in range(max_epochs):
        if (time.time() - start_t) > max_train_time:
            break
        model.train()
        loss_sum = 0
        labels_since_last_log = 0
        iter_since_last_log = 0
        train_mae = 0.0
        tot_loss = 0.0
        epoch_steps = len(train_dataloader)
        i = 0
        try:
            for images, labels in train_dataloader:
                if (time.time() - start_t) > max_train_time:
                    break
                optimizer.zero_grad()
                # Only forward pass should be autocast
                with torch.autocast(device, dtype=dtype):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = model(images).ravel()
                    loss = loss_fn(preds, labels)
                    if max_loss:
                        loss = torch.clamp(loss, min_loss, max_loss)
                        loss[loss != loss] = max_loss
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
                scaler.step(optimizer)
                scaler.update()
                i += 1
                iter_since_last_log += 1
                labels_since_last_log += len(labels)
                loss_sum += loss.data
                this_t = time.time()
                time_diff = this_t - last_log_t
                if time_diff > log_interval:
                    if last_log_t == 0:
                        time_diff = this_t - start_t
                    train_probs = torch.sigmoid(preds).ravel().detach().to("cpu", torch.float32).numpy()
                    train_labels = labels.detach().to("cpu", torch.float32).numpy()
                    train_classes = train_probs > 0.5
                    train_acc = (
                        (((train_classes == 1) & (train_labels == 1)) | ((train_classes == 0) & (train_labels == 0)))
                        .astype(int)
                        .sum()
                    )
                    # Other runtime stats
                    iter_remain = epoch_steps - i
                    epoch_t_remain = (iter_since_last_log / iter_remain) * time_diff
                    eta_val = epoch_t_remain + time.time()
                    eta = dt.fromtimestamp(eta_val, LOCAL_TZ)
                    tot_loss = loss_sum / i
                    log_ts = dt.now().strftime("%H:%M")
                    pct_progress = f"{i / epoch_steps * 100:.1f}"
                    fps_str = f"{labels_since_last_log / time_diff:.1f}"
                    loss_str = f"{loss.data:.3f}"
                    tot_loss_str = f"{tot_loss:.3f}"
                    val = {}
                    val_str = ""
                    if test_dataloader:
                        try:
                            val = test_img(model, test_dataloader, device, dtype=dtype)
                            for k in val_metrics:
                                k_name = k.strip("logits_")
                                val_str = val_str + f"{k_name} {val[k]:.2f}|"
                            for k in val:
                                val[k] = f"{val[k]:.2f}"
                        except Exception as e:
                            KLOGGER.error(f"Error Validating: {str(e)}")
                            val["error"] = str(e)
                    val["model_name"] = model_name
                    val["time"] = log_ts
                    val["fps"] = fps_str
                    val["loss"] = loss_str
                    val["tot_loss"] = tot_loss_str
                    val["eta"] = eta.strftime("%Y_%m_%d_%H_%M")
                    cs.write_json_log(val, save_path, extra_info=model_name)
                    KLOGGER.info(
                        f"{log_ts}|{pct_progress}%|FPS {fps_str} |"
                        + f"ETA {eta.strftime('%H:%M')}|"
                        + f"loss {loss_str}(tot {tot_loss_str})|trainAcc {train_acc:.2f}|"
                        + f"{val_str}"
                    )
                    last_log_t = time.time()
                    labels_since_last_log = 0
                    iter_since_last_log = 0
            if epoch % epoch_save_interval == 0:
                save_model(model, save_path, extra_info=f"{model_name}_{int(tot_loss * 1000):04d}")
        except KeyboardInterrupt:
            KLOGGER.info("Keyboard Interrupt: save model and exit.")
        finally:
            save_model(model, save_path, extra_info=f"{model_name}_{int(tot_loss * 1000):04d}")
            if test_dataloader:
                val = test_img(model, test_dataloader, device, dtype=dtype)
                for k in val:
                    KLOGGER.info(f"{k}: {val[k]:.2f}")


###################
#       VAL       #
###################
def test_img(
    model,
    val_dataloader,
    device=TORCH_DEVICE,
    precision=2,
    dtype=cs.TORCH_DTYPE,
):
    """
    Validate image training
    """
    res = {}
    try:
        start_t = time.time()
        model.eval()
        correct_guesses = 0
        logits = []
        probs = []
        labels = []
        with torch.no_grad():
            for images, this_labels in val_dataloader:
                with torch.autocast(device, dtype=dtype):
                    images = images.to(device, non_blocking=True)
                    this_logits = model(images)
                logits.append(this_logits.ravel().to("cpu", torch.float32, copy=True).numpy())
                labels.append(this_labels.to("cpu", torch.float32, copy=True).numpy())
        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        probs = torch.sigmoid(torch.as_tensor(logits)).to("cpu", torch.float32, copy=True).numpy()
        n_nan_logits = np.count_nonzero(np.isnan(logits))
        logits[np.isnan(logits)] = -9999
        logits_min = np.min(logits)
        logits_max = np.max(logits)
        logits_mean = np.mean(logits)
        logits_q25 = np.quantile(logits, 0.25)
        logits_q50 = np.quantile(logits, 0.5)
        logits_q75 = np.quantile(logits, 0.75)
        probs[np.isnan(probs)] = 0.5
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        pred_classes = np.nan_to_num(probs > 0.5, nan=2.0, posinf=3.0, neginf=-2.0)
        correct_guesses = (
            (((pred_classes == 1) & (labels == 1)) | ((pred_classes == 0) & (labels == 0))).astype(int).sum()
        )
        mae = np.abs(probs - labels).sum() / len(labels)
        eval_res = {
            "accuracy": metrics.accuracy_score(labels, pred_classes),
            "f1": metrics.f1_score(labels, pred_classes, zero_division=np.nan),
            "precision": metrics.precision_score(labels, pred_classes, zero_division=np.nan),
            "recall": metrics.recall_score(labels, pred_classes, zero_division=np.nan),
            "brier_loss": metrics.brier_score_loss(labels, probs),
            "log_loss": metrics.log_loss(labels, probs),
            "roc_auc": metrics.roc_auc_score(labels, probs),
            "correct_guesses": correct_guesses,
            "mae": mae,
            "mean_labels": labels.mean(),
            "mean_pred": probs.mean(),
            "nan_outputs": n_nan_logits,
            "logits_min": logits_min,
            "logits_max": logits_max,
            "logits_mean": logits_mean,
            "logits_q25": logits_q25,
            "logits_q50": logits_q50,
            "logits_q75": logits_q75,
            "eval_samples": len(labels),
            "eval_time": time.time() - start_t,
        }
        res = {k: np.round(eval_res[k], precision) for k in eval_res}
    except Exception as e:
        KLOGGER.error(f"\nERROR:\n---\n{str(e)}\n---{traceback.format_exc()}\n---\n")
    return res
