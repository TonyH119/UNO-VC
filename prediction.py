from pathlib import Path
import gc

import numpy as np
import torch

from VC_uno1d import UNO_VC_60s

ROOT = Path(__file__).resolve().parent

MODEL_FILES = {
    "upward": ROOT / "models" / "upward_model.pt",
    "downward": ROOT / "models" / "downward_model.pt",
}

EXAMPLE_FILES = {
    "upward": ROOT / "example_data" / "upward_example.npz",
    "downward": ROOT / "example_data" / "downward_example.npz",
}

def prepare_input(data_ax, data_vx):
    data_ax = torch.as_tensor(data_ax, dtype=torch.float32)
    data_vx = torch.as_tensor(data_vx, dtype=torch.float32)
    if data_ax.ndim == 2:
        data_ax = data_ax[None, ...]
        data_vx = data_vx[None, ...]

    maxamp_ax = torch.max(torch.abs(data_ax), dim=1, keepdim=True)[0].clamp_min(1e-8)
    maxamp_vx = torch.max(torch.abs(data_vx), dim=1, keepdim=True)[0].clamp_min(1e-8)
    norm_factorax = maxamp_ax.expand_as(data_ax)
    norm_factorvx = maxamp_vx.expand_as(data_vx)

    return torch.cat(
        (
            data_ax / norm_factorax,
            data_vx / norm_factorvx,
            torch.log10(norm_factorax),
            torch.log10(norm_factorvx),
        ),
        dim=2,
    )

def load_trained_model(direction, device):
    if direction not in MODEL_FILES:
        raise ValueError("direction must be 'upward' or 'downward'")

    channel_num = 3
    wav_ch_out = np.arange(channel_num * 2)
    amp_ch_out = np.arange(channel_num * 2, channel_num * 4)
    model = UNO_VC_60s(
        in_width=channel_num * 4 + 1,
        out_width=channel_num * 4,
        width=64,
        pad=0,
        dropout_rate=0.2,
        wav_ch_out=wav_ch_out,
        amp_ch_out=amp_ch_out,
    )

    state_dict = torch.load(MODEL_FILES[direction], map_location="cpu")
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.to(device)
    model.eval()
    return model

def reconstruct_prediction(raw_output, data_ax, data_vx):
    raw_output = np.asarray(raw_output, dtype=np.float32)
    data_ax = np.asarray(data_ax, dtype=np.float32)
    data_vx = np.asarray(data_vx, dtype=np.float32)

    wav_predicted = raw_output[:, :, :6]
    wav_predicted = wav_predicted / (
        np.max(np.abs(wav_predicted), axis=1, keepdims=True) + 1e-5
    )

    input_amplitude = np.concatenate(
        (
            np.log10(np.maximum(np.max(np.abs(data_ax), axis=1), 1e-8)),
            np.log10(np.maximum(np.max(np.abs(data_vx), axis=1), 1e-8)),
        ),
        axis=1,
    )
    predicted_amplitude = input_amplitude + np.mean(raw_output[:, :, 6:12], axis=1)
    physical_prediction = wav_predicted * 10 ** predicted_amplitude[:, None, :]
    return physical_prediction[:, :, :3], physical_prediction[:, :, 3:6]

def run_prediction(direction="upward", device="cuda"):
    if direction not in EXAMPLE_FILES:
        raise ValueError("direction must be 'upward' or 'downward'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    with np.load(EXAMPLE_FILES[direction], allow_pickle=False) as file:
        example = {key: file[key] for key in file.files}

    data_x = prepare_input(
        example["input_acceleration"],
        example["input_velocity"],
    )
    model = load_trained_model(direction, device)
    with torch.no_grad():
        raw_output = model(data_x.to(device)).cpu().numpy()

    predicted_acceleration, predicted_velocity = reconstruct_prediction(
        raw_output,
        example["input_acceleration"],
        example["input_velocity"],
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return example, predicted_acceleration, predicted_velocity, raw_output
