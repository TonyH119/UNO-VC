
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralConv1d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1,modes1 = None):
        super(SpectralConv1d_Uno, self).__init__()

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        if modes1 is not None:
            self.modes1 = modes1
        else:
            self.modes1 = dim1//2
        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1 = None):

        if dim1 is not None:
            self.dim1 = dim1
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x, norm = 'forward')

        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        x = torch.fft.irfft(out_ft, n=self.dim1, norm = 'forward')

        return x

class pointwise_op_1D(nn.Module):

    def __init__(self, in_codim, out_codim,dim1):
        super(pointwise_op_1D,self).__init__()
        self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)

    def forward(self,x, dim1 = None):
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True)
        return x_out

class OperatorBlock_1D(nn.Module):

    def __init__(self, in_codim, out_codim,dim1,modes1, Normalize = True,Non_Lin = True):
        super(OperatorBlock_1D,self).__init__()
        self.conv = SpectralConv1d_Uno(in_codim, out_codim, dim1,modes1)
        self.w = pointwise_op_1D(in_codim, out_codim, dim1)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim),affine=True)

    def forward(self,x, dim1 = None):

        x1_out = self.conv(x,dim1)

        x2_out = self.w(x,dim1)

        x_out = x1_out + x2_out

        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class Output_conv_fc(nn.Module):
    def __init__(self, in_codim, out_codim, dim1,modes1 = None,wav_ch_out=np.array([]),amp_ch_out=np.array([]),out_channel_mid=32,conv1d_dim=100,dropout_rate=0.3):
        super(Output_conv_fc, self).__init__()

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        assert(wav_ch_out.size+amp_ch_out.size==self.out_channels)
        self.wav_ch_out = wav_ch_out
        self.amp_ch_out = amp_ch_out
        self.dropout_rate = dropout_rate

        self.out_channel_fc_wav = out_channel_mid
        self.fc_wav = nn.Linear(self.in_channels, self.out_channel_fc_wav)
        self.dropout_wav = nn.Dropout(self.dropout_rate)
        self.in_channel_fourier_wav = self.in_channels
        self.out_channel_fourier_wav = self.wav_ch_out.size
        self.dim1 = dim1
        if modes1 is not None:
            self.modes1 = modes1
        else:
            self.modes1 = dim1//2
        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(self.in_channel_fourier_wav, self.out_channel_fourier_wav, self.modes1, dtype=torch.cfloat))

        self.out_channel_conv1d = out_channel_mid*2
        self.conv1d1 = nn.Conv1d(in_channels=self.in_channels,out_channels=self.out_channel_conv1d,kernel_size=100,stride=10,padding=9)
        self.dropout_amp = nn.Dropout(self.dropout_rate)
        self.conv1d2 = nn.Conv1d(in_channels=self.out_channel_conv1d,out_channels=self.amp_ch_out.size,kernel_size=30,stride=10,padding=9)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_channel_fc_amp = self.amp_ch_out.size
        self.fc_amp1 = nn.Linear(self.in_channels,out_channel_mid)
        self.fc_amp2 = nn.Linear(out_channel_mid,self.out_channel_fc_amp)
        self.fc_amp3 = nn.Linear(self.out_channel_fc_amp,self.out_channel_fc_amp)

        self.modes2 = 1
        self.in_channel_fourier_amp = self.in_channels
        self.out_channel_fourier_amp = out_channel_mid
        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights2 = nn.Parameter(self.scale * torch.randn(self.in_channel_fourier_amp, self.out_channel_fourier_amp, self.modes2, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1 = None):

        if dim1 is not None:
            self.dim1 = dim1
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x, norm = 'forward')

        out_ft_wav = torch.zeros(batchsize, self.out_channel_fourier_wav,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft_wav[:, :, 1:self.modes1+1] = self.compl_mul1d(x_ft[:, :, 1:self.modes1+1], self.weights1)
        x_out = torch.fft.irfft(out_ft_wav, n=self.dim1, norm = 'forward')

        out_ft_amp = torch.zeros(batchsize, self.out_channel_fourier_amp,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)

        out_ft_amp[:, :,:self.modes2] = self.compl_mul1d(x_ft[:, :, :self.modes2], self.weights2)
        x_amp = torch.fft.irfft(out_ft_amp, n=self.dim1, norm = 'forward')
        x_amp = F.gelu(x_amp)
        x_amp = self.dropout_wav(x_amp)
        x_amp = self.fc_amp2(x_amp.permute(0,2,1))
        x_amp = x_amp.permute(0,2,1)

        x_out = torch.concat((x_out,x_amp),dim=1)

        return x_out

class UNO_VC_60s(nn.Module):

    def __init__(self, in_width,out_width, width,pad = 0, factor = 1,dropout_rate = 0,wav_ch_out=np.array([]),amp_ch_out=np.array([])):
        super(UNO_VC_60s, self).__init__()

        self.in_width = in_width
        self.out_width = out_width
        self.width = width
        self.padding = pad
        self.dropout_rate_fc = dropout_rate
        self.dropout_rate = dropout_rate
        self.wav_ch_out = wav_ch_out
        self.amp_ch_out = amp_ch_out

        self.fc0 = nn.Linear(self.in_width, self.width//2)
        self.dropout_fc0 = nn.Dropout(self.dropout_rate_fc/2)

        self.fc1 = nn.Linear(self.width//2, self.width)
        self.dropout_fc1 = nn.Dropout(self.dropout_rate_fc/2)

        self.conv0 = OperatorBlock_1D(self.width, 2*factor*self.width,3000, 1500)
        self.dropout_c0 = nn.Dropout(self.dropout_rate)

        self.conv1 = OperatorBlock_1D(2*factor*self.width, 4*factor*self.width, 2400, 1200, Normalize = True)
        self.dropout_c1 = nn.Dropout(self.dropout_rate)

        self.conv2 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 2400, 1200)
        self.dropout_c2 = nn.Dropout(self.dropout_rate)

        self.conv4 = OperatorBlock_1D(4*factor*self.width, 2*factor*self.width, 3000, 1200, Normalize = True)
        self.dropout_c4 = nn.Dropout(self.dropout_rate)

        self.conv5 = OperatorBlock_1D(4*factor*self.width, self.width, 6001,1500)
        self.dropout_c5 = nn.Dropout(self.dropout_rate)

        self.fc2 = nn.Linear(2*self.width, self.width//2)
        self.dropout_fc2 = nn.Dropout(self.dropout_rate_fc)

        self.output = Output_conv_fc(2*self.width, self.out_width, 6001,1500,out_channel_mid=self.width,wav_ch_out=wav_ch_out,amp_ch_out=amp_ch_out)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, grid), dim=-1)

        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)
        x_fc0 = self.dropout_fc0(x_fc0)

        x_fc1 = self.fc1(x_fc0)
        x_fc1 = F.gelu(x_fc1)
        x_fc1 = self.dropout_fc1(x_fc1)
        x_fc1 = x_fc1.permute(0, 2, 1)
        scale = math.ceil(x_fc1.shape[-1]/85)
        x_fc1 = F.pad(x_fc1, [0,scale*self.padding])

        x_c0 = self.conv0(x_fc1)
        x_c0 = self.dropout_c0(x_c0)

        x_c1 = self.conv1(x_c0)
        x_c1 = self.dropout_c1(x_c1)

        x_c2 = self.conv2(x_c1)
        x_c2 = self.dropout_c2(x_c2)

        x_c4 = self.conv4(x_c2)
        x_c4 = self.dropout_c4(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x_c5 = self.conv5(x_c4)
        x_c5 = self.dropout_c5(x_c5)
        x_c5 = torch.cat([x_c5, x_fc1], dim=1)

        x_out = self.output(x_c5)

        x_out = x_out.permute(0, 2, 1)
        x_out = x_out.contiguous()

        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

def butter_bandpass_sos(freqmin=1.0, freqmax=20.0, fs=100.0, order=4):
    from scipy import signal

    if freqmin <= 0 or freqmax <= freqmin or freqmax >= fs / 2:
        raise ValueError('Bandpass frequencies must satisfy 0 < freqmin < freqmax < fs/2.')
    return signal.butter(order, [freqmin, freqmax], btype='bandpass', fs=fs, output='sos')

def _sosfiltfilt_padlen(sos):
    sos = np.asarray(sos)
    return 3 * (2 * sos.shape[0] + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))

def torch_sosfiltfilt_response(n, sos, fs=100.0, device=None, dtype=torch.float32):
    sos = torch.as_tensor(sos, device=device, dtype=dtype)
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs).to(device=device, dtype=dtype)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    zinv = torch.exp(freqs.to(complex_dtype) * (-2j * math.pi / fs))
    h = torch.ones_like(zinv)

    for sec in sos:
        b0, b1, b2, a0, a1, a2 = sec.to(complex_dtype)
        h = h * (b0 + b1 * zinv + b2 * zinv ** 2) / (a0 + a1 * zinv + a2 * zinv ** 2)

    return (h.real ** 2 + h.imag ** 2).to(dtype)

def torch_bandpass_sosfiltfilt(x, dim=1, sos=None, freqmin=1.0, freqmax=20.0, fs=100.0, order=4, response=None, padlen=None):

    if sos is None:
        sos = butter_bandpass_sos(freqmin=freqmin, freqmax=freqmax, fs=fs, order=order)
    if padlen is None:
        padlen = _sosfiltfilt_padlen(sos)

    dim = dim if dim >= 0 else x.dim() + dim
    x_filter = x.movedim(dim, -1)
    n_original = x_filter.shape[-1]

    if n_original <= 1:
        return x

    if n_original > padlen + 1:
        left = 2 * x_filter[..., :1] - torch.flip(x_filter[..., 1:padlen + 1], dims=[-1])
        right = 2 * x_filter[..., -1:] - torch.flip(x_filter[..., -padlen - 1:-1], dims=[-1])
        x_filter = torch.cat((left, x_filter, right), dim=-1)
    else:
        padlen = 0

    n = x_filter.shape[-1]
    if response is None:
        response = torch_sosfiltfilt_response(n, sos, fs=fs, device=x.device, dtype=x.dtype)

    response_shape = [1] * x_filter.dim()
    response_shape[-1] = response.shape[0]
    response = response.reshape(response_shape)

    x_ft = torch.fft.rfft(x_filter, dim=-1)
    y_filter = torch.fft.irfft(x_ft * response, n=n, dim=-1)
    if padlen > 0:
        y_filter = y_filter[..., padlen:-padlen]
    return y_filter.movedim(-1, dim)

def torch_bandpass_1_20hz_sosfiltfilt(x, dim=1):
    return torch_bandpass_sosfiltfilt(x, dim=dim, freqmin=1.0, freqmax=20.0, fs=100.0, order=4)

class UNO_VC_Loss(object):
    def __init__(self,d=2,p=2,nchannel=12,wav_ch=np.arange(0,6),amp_ch=np.arange(6,12),size_average=True, reduction=True, bandpass_loss=False,
                 freqmin=1.0, freqmax=20.0, fs=100.0, bandpass_order=4):
        super(UNO_VC_Loss, self).__init__()
        assert d > 0 and p > 0
        self.nchannel = nchannel
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.bandpass_loss = bandpass_loss
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.fs = fs
        self.bandpass_order = bandpass_order
        self.bandpass_sos = None
        self.bandpass_padlen = None
        self.bandpass_response_cache = {}
        if self.bandpass_loss:
            self.bandpass_sos = butter_bandpass_sos(freqmin, freqmax, fs, bandpass_order)
            self.bandpass_padlen = _sosfiltfilt_padlen(self.bandpass_sos)

        self.wav_ch = wav_ch
        self.amp_ch = amp_ch
        self.wav_ch_num = len(wav_ch)
        self.amp_ch_num = len(amp_ch)
        self.wav_loss_type = 'cc'
        self.variance_amp1 = 0.01 * 2
        self.variance_amp2 = 0.02 * 2
        self.abs_std_amp1 = np.sqrt(self.variance_amp1) * np.sqrt(2.0/np.pi)
        self.abs_std_amp2 = np.sqrt(self.variance_amp2) * np.sqrt(2.0/np.pi)

    def _bandpass_waveform(self, x):
        if self.bandpass_sos is None:
            self.bandpass_sos = butter_bandpass_sos(self.freqmin, self.freqmax, self.fs, self.bandpass_order)
            self.bandpass_padlen = _sosfiltfilt_padlen(self.bandpass_sos)

        padlen = self.bandpass_padlen if x.shape[1] > self.bandpass_padlen + 1 else 0
        n_filter = x.shape[1] + 2 * padlen
        cache_key = (str(x.device), x.dtype, n_filter)
        response = self.bandpass_response_cache.get(cache_key)
        if response is None:
            response = torch_sosfiltfilt_response(n_filter, self.bandpass_sos, fs=self.fs, device=x.device, dtype=x.dtype)
            self.bandpass_response_cache[cache_key] = response

        return torch_bandpass_sosfiltfilt(x, dim=1, sos=self.bandpass_sos, fs=self.fs,
                                          response=response, padlen=self.bandpass_padlen)

    def rel(self, x, y, prcWav, prcAmp):

        num_examples = x.size()[0]
        x_recover = x.reshape(num_examples,-1,self.nchannel)
        y_recover = y.reshape(num_examples,-1,self.nchannel)

        x_wav_all = x_recover[:,:,self.wav_ch]
        y_wav_all = y_recover[:,:,self.wav_ch]
        if self.bandpass_loss:
            x_wav_all = self._bandpass_waveform(x_wav_all)
        x_wav1 = x_wav_all[:,:,:self.wav_ch_num//2]
        x_wav2 = x_wav_all[:,:,self.wav_ch_num//2:]
        y_wav1 = y_wav_all[:,:,:self.wav_ch_num//2]
        y_wav2 = y_wav_all[:,:,self.wav_ch_num//2:]
        if self.wav_loss_type == 'diff':
            diff1 = x_wav1 - y_wav1
            diff2 = x_wav2 - y_wav2
            diff_norms_wav1 = torch.norm(diff1, self.p, 1)
            diff_norms_wav2 = torch.norm(diff2, self.p, 1)
            y_norms_wav1 = torch.norm(y_wav1, self.p, 1)
            y_norms_wav2 = torch.norm(y_wav2, self.p, 1)
            loss_wav1 = diff_norms_wav1 / (y_norms_wav1+1e-8)
            loss_wav2 = diff_norms_wav2 / (y_norms_wav2+1e-8)

        elif self.wav_loss_type == 'cc':
            cc1 = torch.sum(x_wav1 * y_wav1,dim=1)
            cc2 = torch.sum(x_wav2 * y_wav2,dim=1)
            loss_wav1 = (1 - cc1 / ((torch.norm(y_wav1, self.p, 1)+1e-8)*(torch.norm(x_wav1, self.p, 1)+1e-8)))*2
            loss_wav2 = (1 - cc2 / ((torch.norm(y_wav2, self.p, 1)+1e-8)*(torch.norm(x_wav2, self.p, 1)+1e-8)))*2

        prcWav /= self.wav_ch_num

        x_amp_all = x_recover[:,:,self.amp_ch]
        y_amp_all = y_recover[:,:,self.amp_ch]
        x_amp1 = x_amp_all[:,:,:self.amp_ch_num//2]
        x_amp2 = x_amp_all[:,:,self.amp_ch_num//2:]
        y_amp1 = y_amp_all[:,:,:self.amp_ch_num//2]
        y_amp2 = y_amp_all[:,:,self.amp_ch_num//2:]
        diff1 = x_amp1 - y_amp1
        diff2 = x_amp2 - y_amp2

        diff_norms_amp1 = torch.abs(torch.mean(diff1,dim=1))
        diff_norms_amp2 = torch.abs(torch.mean(diff2,dim=1))
        loss_amp1 = diff_norms_amp1 / self.abs_std_amp1
        loss_amp2 = diff_norms_amp2 / self.abs_std_amp2

        prcAmp /= self.amp_ch_num
        if self.reduction:
            if self.size_average:
                loss_wav1 = torch.mean(loss_wav1) * prcWav
                loss_wav2 = torch.mean(loss_wav2) * prcWav
                loss_amp1 = torch.mean(loss_amp1) * prcAmp
                loss_amp2 = torch.mean(loss_amp2) * prcAmp
            else:
                loss_wav1 = torch.sum(loss_wav1) * prcWav
                loss_wav2 = torch.sum(loss_wav2) * prcWav
                loss_amp1 = torch.sum(loss_amp1) * prcAmp
                loss_amp2 = torch.sum(loss_amp2) * prcAmp
        else:
            loss_wav1 = torch.sum(loss_wav1,dim=1) * prcWav
            loss_wav2 = torch.sum(loss_wav2,dim=1) * prcWav
            loss_amp1 = torch.sum(loss_amp1,dim=1) * prcAmp
            loss_amp2 = torch.sum(loss_amp2,dim=1) * prcAmp
        return (loss_wav1+loss_amp1+loss_wav2+loss_amp2),(loss_amp1+loss_amp2)

    def __call__(self, x, y, prcWav, prcAmp):
        return self.rel(x, y, prcWav, prcAmp)

class UNO_VC_transfer_Loss(object):
    def __init__(self,d=2,p=2,nchannel=12,wav_ch=np.arange(0,6),amp_ch=np.arange(6,12),size_average=True, reduction=True, bandpass_loss=False,
                 freqmin=1.0, freqmax=20.0, fs=100.0, bandpass_order=4):
        super(UNO_VC_transfer_Loss, self).__init__()
        assert d > 0 and p > 0
        self.nchannel = nchannel
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.bandpass_loss = bandpass_loss
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.fs = fs
        self.bandpass_order = bandpass_order
        self.bandpass_sos = None
        self.bandpass_padlen = None
        self.bandpass_response_cache = {}
        if self.bandpass_loss:
            self.bandpass_sos = butter_bandpass_sos(freqmin, freqmax, fs, bandpass_order)
            self.bandpass_padlen = _sosfiltfilt_padlen(self.bandpass_sos)

        self.wav_ch = wav_ch
        self.amp_ch = amp_ch
        self.wav_ch_num = len(wav_ch)
        self.amp_ch_num = len(amp_ch)
        self.wav_loss_type = 'cc'
        self.variance_amp1 = 0.01 * 2
        self.variance_amp2 = 0.01 * 2
        self.abs_std_amp1 = np.sqrt(self.variance_amp1) * np.sqrt(2.0/np.pi)
        self.abs_std_amp2 = np.sqrt(self.variance_amp2) * np.sqrt(2.0/np.pi)

    def _bandpass_waveform(self, x):
        if self.bandpass_sos is None:
            self.bandpass_sos = butter_bandpass_sos(self.freqmin, self.freqmax, self.fs, self.bandpass_order)
            self.bandpass_padlen = _sosfiltfilt_padlen(self.bandpass_sos)

        padlen = self.bandpass_padlen if x.shape[1] > self.bandpass_padlen + 1 else 0
        n_filter = x.shape[1] + 2 * padlen
        cache_key = (str(x.device), x.dtype, n_filter)
        response = self.bandpass_response_cache.get(cache_key)
        if response is None:
            response = torch_sosfiltfilt_response(n_filter, self.bandpass_sos, fs=self.fs, device=x.device, dtype=x.dtype)
            self.bandpass_response_cache[cache_key] = response

        return torch_bandpass_sosfiltfilt(x, dim=1, sos=self.bandpass_sos, fs=self.fs,
                                          response=response, padlen=self.bandpass_padlen)

    def rel(self, x, y, prcWav, prcAmp):

        num_examples = x.size()[0]
        x_recover = x.reshape(num_examples,-1,self.nchannel)
        y_recover = y.reshape(num_examples,-1,self.nchannel)

        x_wav_all = x_recover[:,:,self.wav_ch]
        y_wav_all = y_recover[:,:,self.wav_ch]
        if self.bandpass_loss:
            x_wav_all = self._bandpass_waveform(x_wav_all)
        x_wav1 = x_wav_all[:,:,:self.wav_ch_num//2]
        x_wav2 = x_wav_all[:,:,self.wav_ch_num//2:]
        y_wav1 = y_wav_all[:,:,:self.wav_ch_num//2]
        y_wav2 = y_wav_all[:,:,self.wav_ch_num//2:]
        if self.wav_loss_type == 'diff':
            diff1 = x_wav1 - y_wav1
            diff2 = x_wav2 - y_wav2
            diff_norms_wav1 = torch.norm(diff1, self.p, 1)
            diff_norms_wav2 = torch.norm(diff2, self.p, 1)
            y_norms_wav1 = torch.norm(y_wav1, self.p, 1)
            y_norms_wav2 = torch.norm(y_wav2, self.p, 1)
            loss_wav1 = diff_norms_wav1 / (y_norms_wav1+1e-8)
            loss_wav2 = diff_norms_wav2 / (y_norms_wav2+1e-8)

        elif self.wav_loss_type == 'cc':
            cc1 = torch.sum(x_wav1 * y_wav1,dim=1)
            cc2 = torch.sum(x_wav2 * y_wav2,dim=1)
            loss_wav1 = (1 - cc1 / ((torch.norm(y_wav1, self.p, 1)+1e-8)*(torch.norm(x_wav1, self.p, 1)+1e-8)))*2
            loss_wav2 = (1 - cc2 / ((torch.norm(y_wav2, self.p, 1)+1e-8)*(torch.norm(x_wav2, self.p, 1)+1e-8)))*2

        prcWav /= self.wav_ch_num

        x_amp_all = x_recover[:,:,self.amp_ch]
        y_amp_all = y_recover[:,:,self.amp_ch]
        x_amp1 = x_amp_all[:,:,:self.amp_ch_num//2]
        x_amp2 = x_amp_all[:,:,self.amp_ch_num//2:]
        y_amp1 = y_amp_all[:,:,:self.amp_ch_num//2]
        y_amp2 = y_amp_all[:,:,self.amp_ch_num//2:]
        diff1 = x_amp1 - y_amp1
        diff2 = x_amp2 - y_amp2

        diff_norms_amp1 = torch.abs(torch.mean(diff1,dim=1))
        diff_norms_amp2 = torch.abs(torch.mean(diff2,dim=1))
        loss_amp1 = diff_norms_amp1 / self.abs_std_amp1
        loss_amp2 = diff_norms_amp2 / self.abs_std_amp2

        prcAmp /= self.amp_ch_num
        if self.reduction:
            if self.size_average:
                loss_wav1 = torch.mean(loss_wav1) * prcWav
                loss_wav2 = torch.mean(loss_wav2) * prcWav
                loss_amp1 = torch.mean(loss_amp1) * prcAmp
                loss_amp2 = torch.mean(loss_amp2) * prcAmp
            else:
                loss_wav1 = torch.sum(loss_wav1) * prcWav
                loss_wav2 = torch.sum(loss_wav2) * prcWav
                loss_amp1 = torch.sum(loss_amp1) * prcAmp
                loss_amp2 = torch.sum(loss_amp2) * prcAmp
        else:
            loss_wav1 = torch.sum(loss_wav1,dim=1) * prcWav
            loss_wav2 = torch.sum(loss_wav2,dim=1) * prcWav
            loss_amp1 = torch.sum(loss_amp1,dim=1) * prcAmp
            loss_amp2 = torch.sum(loss_amp2,dim=1) * prcAmp
        return (loss_wav1+loss_amp1+loss_wav2+loss_amp2),(loss_amp1+loss_amp2)

    def __call__(self, x, y, prcWav, prcAmp):
        return self.rel(x, y, prcWav, prcAmp)