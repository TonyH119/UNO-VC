from pathlib import Path
import gc
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import load_data
from train_VC import train_model
from VC_uno1d import UNO_VC_60s

ROOT = Path(__file__).resolve().parent

# Change only these settings for a normal run.
DIRECTION = "downward"
DATA_FILES = {
    "upward": ROOT / "data" / "Clean_upward_tapered_05-00_ZEN3CH_60s.mat",
    "downward": ROOT / "data" / "Clean_downward_untapered_00-05_ZEN3CH_60s.mat",
}
DEVICE = "cuda"
N_FOLDS = 10
FOLDS_TO_RUN = 10

FS = 100
TIME_LENGTH = 60
DECIMATE_RATE = 1
BATCH_SIZE = 32
WIDTH = 64
EPOCHS = 150
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.2
SCHEDULER_STEP = 5
SCHEDULER_GAMMA = 0.9
WEIGHT_DECAY = 0.01

def main():
    if DIRECTION not in DATA_FILES:
        raise ValueError("DIRECTION must be 'upward' or 'downward'")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(10001)
    random.seed(10001)
    data_file = DATA_FILES[DIRECTION]
    result_folder = ROOT / "results" / ("base_" + DIRECTION)
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
    expected_samples = int(FS * (TIME_LENGTH / DECIMATE_RATE) + 1)
    if sample_count != expected_samples:
        raise ValueError("Unexpected waveform sample count")

    logPGA_x = np.log10(PGA_x.reshape(n_data).numpy())
    logPGA_y = np.log10(PGA_y.reshape(n_data).numpy())
    logPGV_x = np.log10(PGV_x.reshape(n_data).numpy())
    logPGV_y = np.log10(PGV_y.reshape(n_data).numpy())
    logPGA_diff = logPGA_y - logPGA_x
    logPGV_diff = logPGV_y - logPGV_x

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

    wav_ch_out = np.arange(channel_num * 2)
    amp_ch_out = np.arange(channel_num * 2, channel_num * 4)
    nchannel_in = channel_num * 4
    nchannel_out = channel_num * 4

    indices = list(range(n_data))
    random.shuffle(indices)
    fold_no = np.zeros(n_data, dtype=int)
    for fold in range(N_FOLDS):
        start = int(n_data * fold / N_FOLDS)
        stop = int(n_data * (fold + 1) / N_FOLDS)
        fold_no[[value for position, value in enumerate(indices) if start <= position < stop]] = fold

    np.savez(
        result_folder / "hyper_parameters.npz",
        data_file=str(data_file),
        n_data=n_data,
        batch_size=BATCH_SIZE,
        width=WIDTH,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE,
        scheduler_step=SCHEDULER_STEP,
        scheduler_gamma=SCHEDULER_GAMMA,
        weight_decay=WEIGHT_DECAY,
        indices=indices,
        n_folds=N_FOLDS,
        fold_no=fold_no,
    )
    np.savez(
        result_folder / "logPGA_PGV.npz",
        logPGA_x=logPGA_x,
        logPGA_y=logPGA_y,
        logPGV_x=logPGV_x,
        logPGV_y=logPGV_y,
        logPGA_diff=logPGA_diff,
        logPGV_diff=logPGV_diff,
        logPGA_diff_median=np.median(logPGA_diff),
        logPGV_diff_median=np.median(logPGV_diff),
    )

    start_time = time.time()
    for fold in range(min(FOLDS_TO_RUN, N_FOLDS)):
        fold_folder = result_folder / ("fold_" + str(fold))
        fold_folder.mkdir(exist_ok=True)
        val_start = int(n_data * fold / N_FOLDS)
        val_stop = int(n_data * (fold + 1) / N_FOLDS)
        val_ind = [value for position, value in enumerate(indices) if val_start <= position < val_stop]
        train_ind = [value for position, value in enumerate(indices) if position < val_start or position >= val_stop]

        train_x, val_x = data_x[train_ind], data_x[val_ind]
        train_y, val_y = data_y[train_ind], data_y[val_ind]
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

        np.savez(
            fold_folder / "fold_parameters.npz",
            train_ind=train_ind,
            val_ind=val_ind,
        )

        model = UNO_VC_60s(
            in_width=nchannel_in + 1,
            out_width=nchannel_out,
            width=WIDTH,
            pad=0,
            dropout_rate=DROPOUT_RATE,
            wav_ch_out=wav_ch_out,
            amp_ch_out=amp_ch_out,
        ).to(DEVICE)

        model_file_best = fold_folder / "model_best.pt"
        model_file_final = fold_folder / "model_final.pt"
        loss_file = fold_folder / "loss.npz"
        train_model(
            model,
            train_loader,
            val_loader,
            val_loader,
            len(train_ind),
            len(val_ind),
            len(val_ind),
            nchannel_in,
            nchannel_out,
            sample_count,
            model_file_best,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
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
        plt.legend()
        plt.tight_layout()
        plt.savefig(fold_folder / "loss_curve.png")
        plt.close()

        del model, train_x, train_y, val_x, val_y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Finished in", (time.time() - start_time) / 3600, "hours")

if __name__ == "__main__":
    main()
