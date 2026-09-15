from pathlib import Path
import gc
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import load_data
from transfer_VC import transfer_model
from VC_uno1d import UNO_VC_60s

ROOT = Path(__file__).resolve().parent

# Change the direction and target data path for a normal run.
DIRECTION = "upward"
DATA_FILES = {
    "upward": ROOT / "data" / "transfer_upward.mat",
    "downward": ROOT / "data" / "transfer_downward.mat",
}
MODEL_FILES = {
    "upward": ROOT / "models" / "upward_model.pt",
    "downward": ROOT / "models" / "downward_model.pt",
}
SOURCE_LOG_PGA_DIFF = {
    "upward": 0.7758503556251526,
    "downward": -0.7784769535064697,
}
SOURCE_LOG_PGV_DIFF = {
    "upward": 0.794963002204895,
    "downward": -0.7973196506500244,
}

DEVICE = "cuda"
FS = 100
TIME_LENGTH = 60
DECIMATE_RATE = 1

MAXIMUM_TRAIN_NUM = 1000
TRAIN_NUM = 100
MAXIMUM_VAL_NUM = 500
VAL_NUM = 50

BATCH_SIZE = 32
WIDTH = 64
EPOCHS = 150
LEARNING_RATE = 5e-4
DROPOUT_RATE = 0.2
SCHEDULER_STEP = 5
SCHEDULER_GAMMA = 0.9
WEIGHT_DECAY = 0.01
FINETUNE_SCOPE = "decoder_output"
TRANSFER_LR_STRATEGY = "layerwise"
LAYERWISE_DECAY = 0.5

def main():
    if DIRECTION not in DATA_FILES:
        raise ValueError("DIRECTION must be 'upward' or 'downward'")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(10001)
    random.seed(10001)
    data_file = DATA_FILES[DIRECTION]
    result_folder = ROOT / "results" / ("transfer_" + DIRECTION)
    result_folder.mkdir(parents=True, exist_ok=True)

    (
        data_ax,
        data_ay,
        data_vx,
        data_vy,
        PGA_x,
        PGA_y,
        PGV_x,
        PGV_y,
    ) = load_data(FS, TIME_LENGTH, DECIMATE_RATE, data_file)

    n_data, sample_count, channel_num = data_ax.shape
    if n_data <= MAXIMUM_TRAIN_NUM + MAXIMUM_VAL_NUM:
        raise ValueError("Target dataset is too small for the configured split pools")

    logPGA_diff = np.log10(PGA_y.reshape(n_data).numpy()) - np.log10(PGA_x.reshape(n_data).numpy())
    logPGV_diff = np.log10(PGV_y.reshape(n_data).numpy()) - np.log10(PGV_x.reshape(n_data).numpy())

    maxamp_ax = torch.max(torch.abs(data_ax), dim=1, keepdim=True)[0].clamp_min(1e-8)
    maxamp_ay = torch.max(torch.abs(data_ay), dim=1, keepdim=True)[0].clamp_min(1e-8)
    maxamp_vx = torch.max(torch.abs(data_vx), dim=1, keepdim=True)[0].clamp_min(1e-8)
    maxamp_vy = torch.max(torch.abs(data_vy), dim=1, keepdim=True)[0].clamp_min(1e-8)

    norm_factorax = maxamp_ax.expand(n_data, sample_count, channel_num)
    norm_factoray = maxamp_ay.expand(n_data, sample_count, channel_num)
    norm_factorvx = maxamp_vx.expand(n_data, sample_count, channel_num)
    norm_factorvy = maxamp_vy.expand(n_data, sample_count, channel_num)

    data_x = torch.cat(
        (
            data_ax / norm_factorax,
            data_vx / norm_factorvx,
            torch.log10(norm_factorax),
            torch.log10(norm_factorvx),
        ),
        dim=2,
    )
    data_y = torch.cat(
        (
            data_ay / norm_factoray,
            data_vy / norm_factorvy,
            torch.log10(norm_factoray) - torch.log10(norm_factorax),
            torch.log10(norm_factorvy) - torch.log10(norm_factorvx),
        ),
        dim=2,
    )

    indices = list(range(n_data))
    random.shuffle(indices)
    train_ind = indices[:TRAIN_NUM]
    val_ind = indices[MAXIMUM_TRAIN_NUM : MAXIMUM_TRAIN_NUM + VAL_NUM]
    test_ind = indices[MAXIMUM_TRAIN_NUM + MAXIMUM_VAL_NUM :]
    target_log_pga_diff = float(np.median(logPGA_diff[train_ind]))
    target_log_pgv_diff = float(np.median(logPGV_diff[train_ind]))

    train_x, val_x, test_x = data_x[train_ind], data_x[val_ind], data_x[test_ind]
    train_y, val_y, test_y = data_y[train_ind], data_y[val_ind], data_y[test_ind]
    del data_x, data_y

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_x, val_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    wav_ch_out = np.arange(channel_num * 2)
    amp_ch_out = np.arange(channel_num * 2, channel_num * 4)
    nchannel_in = channel_num * 4
    nchannel_out = channel_num * 4
    model = UNO_VC_60s(
        in_width=nchannel_in + 1,
        out_width=nchannel_out,
        width=WIDTH,
        pad=0,
        dropout_rate=DROPOUT_RATE,
        wav_ch_out=wav_ch_out,
        amp_ch_out=amp_ch_out,
    ).to(DEVICE)

    state_dict = torch.load(MODEL_FILES[DIRECTION], map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    with torch.no_grad():
        model.output.fc_amp2.bias[:channel_num] -= (
            SOURCE_LOG_PGA_DIFF[DIRECTION] - target_log_pga_diff
        )
        model.output.fc_amp2.bias[channel_num:] -= (
            SOURCE_LOG_PGV_DIFF[DIRECTION] - target_log_pgv_diff
        )

    np.savez(
        result_folder / "transfer_parameters.npz",
        data_file=str(data_file),
        reference_model=str(MODEL_FILES[DIRECTION]),
        train_ind=train_ind,
        val_ind=val_ind,
        test_ind=test_ind,
        source_log_pga_diff=SOURCE_LOG_PGA_DIFF[DIRECTION],
        source_log_pgv_diff=SOURCE_LOG_PGV_DIFF[DIRECTION],
        target_log_pga_diff=target_log_pga_diff,
        target_log_pgv_diff=target_log_pgv_diff,
        finetune_scope=FINETUNE_SCOPE,
        transfer_lr_strategy=TRANSFER_LR_STRATEGY,
        layerwise_decay=LAYERWISE_DECAY,
    )

    model_file_best = result_folder / "model_best.pt"
    model_file_final = result_folder / "model_final.pt"
    loss_file = result_folder / "loss.npz"
    transfer_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        len(train_ind),
        len(val_ind),
        len(test_ind),
        nchannel_in,
        nchannel_out,
        sample_count,
        model_file_best,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        training_mode="transfer",
        finetune_scope=FINETUNE_SCOPE,
        transfer_lr_strategy=TRANSFER_LR_STRATEGY,
        layerwise_decay=LAYERWISE_DECAY,
        scheduler_step=SCHEDULER_STEP,
        scheduler_gamma=SCHEDULER_GAMMA,
        device=DEVICE,
        weight_decay=WEIGHT_DECAY,
        dropout_rate=DROPOUT_RATE,
        model_file_final=model_file_final,
        loss_file=loss_file,
        wav_ch_out=wav_ch_out,
        amp_ch_out=amp_ch_out,
        bandpass_loss=True,
        freqmin=1.0,
        freqmax=20.0,
        fs=FS,
        bandpass_order=4,
    )

    losses = np.load(loss_file)
    plt.figure(figsize=(10, 6))
    plt.plot(losses["train_loss"], label="train")
    plt.plot(losses["val_loss"], label="validation")
    plt.plot(losses["test_loss"], label="test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_folder / "loss_curve.png")
    plt.close()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
