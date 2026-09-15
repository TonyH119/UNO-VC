import h5py
import numpy as np
import scipy.io
import torch

class MatReader:
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        self.file_path = file_path
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except (NotImplementedError, OSError, ValueError):
            self.data = h5py.File(self.file_path, "r")
            self.old_mat = False

    def read_field(self, field):
        value = self.data[field]
        if not self.old_mat:
            value = value[()]
            value = np.transpose(value, axes=range(value.ndim - 1, -1, -1))
        if self.to_float:
            value = value.astype(np.float32)
        if self.to_torch:
            value = torch.from_numpy(value)
            if self.to_cuda:
                value = value.cuda()
        return value

def load_data(fs, time_length, decimate_rate, file_path):
    sample_count = int(fs * (time_length / decimate_rate) + 1)
    print("resolution S", sample_count)
    reader = MatReader(file_path)

    data_ax = reader.read_field("ZEN3CHax")[:, ::decimate_rate, :]
    data_ay = reader.read_field("ZEN3CHay")[:, ::decimate_rate, :]
    data_vx = reader.read_field("ZEN3CHvx")[:, ::decimate_rate, :]
    data_vy = reader.read_field("ZEN3CHvy")[:, ::decimate_rate, :]
    PGA_x = reader.read_field("PGAx")[:, ::decimate_rate, :]
    PGA_y = reader.read_field("PGAy")[:, ::decimate_rate, :]
    PGV_x = reader.read_field("PGVx")[:, ::decimate_rate, :]
    PGV_y = reader.read_field("PGVy")[:, ::decimate_rate, :]

    return data_ax, data_ay, data_vx, data_vy, PGA_x, PGA_y, PGV_x, PGV_y
