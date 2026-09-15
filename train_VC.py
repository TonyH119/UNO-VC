import numpy as np
import torch
import torch.nn as nn
from timeit import default_timer
from VC_uno1d import UNO_VC_Loss

def _shift_waveform_channels(wav, shifts):
    shifted = torch.zeros_like(wav)
    for i, shift in enumerate(shifts.tolist()):
        if shift > 0:
            shifted[i, shift:, :] = wav[i, :-shift, :]
        elif shift < 0:
            shifted[i, :shift, :] = wav[i, -shift:, :]
        else:
            shifted[i] = wav[i]
    return shifted

def _random_augment_batch(x, y, wav_ch_out, amp_ch_out, max_shift=50, amp_perturb=0.05, noise_level=0.05,
                          max_rotation_deg=0, rotation_probability=0.5):
    x = x.clone()
    y = y.clone()

    n_wav = wav_ch_out.size
    n_amp = amp_ch_out.size
    eps = 1e-8

    if max_rotation_deg > 0 and rotation_probability > 0:
            if n_wav != 6 or n_amp != 6:
                raise ValueError('Horizontal rotation expects 3 acceleration and 3 velocity channels.')

            angle = (
                2 * torch.rand(x.shape[0], 1, 1, device=x.device) - 1
            ) * (max_rotation_deg * torch.pi / 180.0)
            rotate_event = (
                torch.rand(x.shape[0], 1, 1, device=x.device)
                < rotation_probability
            )
            angle = angle * rotate_event
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)

            def rotate_horizontal(waveform):
                east = waveform[:, :, 1:2]
                north = waveform[:, :, 2:3]
                return torch.cat((
                    waveform[:, :, 0:1],
                    cos_angle * east - sin_angle * north,
                    sin_angle * east + cos_angle * north,
                ), dim=2)

            for channel_start in (0, 3):
                channel_stop = channel_start + 3
                amp_start = n_wav + channel_start
                amp_stop = amp_start + 3

                x_log_amp = x[:, :1, amp_start:amp_stop]
                y_log_diff = y[:, :1, amp_start:amp_stop]
                x_waveform = x[:, :, channel_start:channel_stop] * torch.pow(10.0, x_log_amp)
                y_waveform = y[:, :, channel_start:channel_stop] * torch.pow(
                    10.0, x_log_amp + y_log_diff
                )

                x_waveform = rotate_horizontal(x_waveform)
                y_waveform = rotate_horizontal(y_waveform)
                x_max = torch.amax(torch.abs(x_waveform), dim=1, keepdim=True).clamp_min(eps)
                y_max = torch.amax(torch.abs(y_waveform), dim=1, keepdim=True).clamp_min(eps)

                x[:, :, channel_start:channel_stop] = x_waveform / x_max
                y[:, :, channel_start:channel_stop] = y_waveform / y_max
                x[:, :, amp_start:amp_stop] = torch.log10(x_max)
                y[:, :, amp_start:amp_stop] = torch.log10(y_max) - torch.log10(x_max)

    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), device=x.device)
        x[:, :, :n_wav] = _shift_waveform_channels(x[:, :, :n_wav], shifts)
        y[:, :, :n_wav] = _shift_waveform_channels(y[:, :, :n_wav], shifts)

    if amp_perturb > 0 and n_amp > 0:
        amp_scale = 1 + (2 * torch.rand(x.shape[0], 1, 1, device=x.device) - 1) * amp_perturb
        amp_shift = torch.log10(amp_scale)
        x[:, :, n_wav:n_wav + n_amp] = x[:, :, n_wav:n_wav + n_amp] + amp_shift

    if noise_level > 0 and n_wav > 0:
        noise_scale = noise_level * torch.rand(x.shape[0], 1, n_wav, device=x.device)
        x[:, :, :n_wav] = x[:, :, :n_wav] + noise_scale * torch.randn_like(x[:, :, :n_wav])
        max_abs = torch.amax(torch.abs(x[:, :, :n_wav]), dim=1, keepdim=True).clamp_min(eps)
        x[:, :, :n_wav] = x[:, :, :n_wav] / max_abs

    return x, y

def train_model(model,train_loader, val_loader, test_loader,ntrain,nval,ntest,nchannel_in,nchannel_out,s,model_file_best,step=1,batch_size=20,epochs=150,learning_rate= 0.001,\
scheduler_step= 50,scheduler_gamma= 0.5,device = 'cuda', weight_decay = 1e-3,dropout_rate = 0,\
    model_file_final=None,loss_file=None,wav_ch_out=np.array([]),amp_ch_out=np.array([]),bandpass_loss=False,\
    freqmin=1.0,freqmax=20.0,fs=100.0,bandpass_order=4):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,amsgrad = False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    best_error = 99

    myloss = UNO_VC_Loss(nchannel=nchannel_out,wav_ch=wav_ch_out,amp_ch=amp_ch_out,size_average=False,bandpass_loss=bandpass_loss,\
                         freqmin=freqmin,freqmax=freqmax,fs=fs,bandpass_order=bandpass_order)
    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    last_save = 0

    train_amp_loss = np.zeros(epochs)
    val_amp_loss = np.zeros(epochs)

    prc=0.5*np.ones(epochs)
    def custom_weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(custom_weights_init)

    for ep in range(epochs):
        prcWav = prc[ep]
        prcAmp = 1- prcWav
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_amp = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y = _random_augment_batch(x, y, wav_ch_out, amp_ch_out)

            batch_size = x.shape[0]
            optimizer.zero_grad()

            out = model(x).reshape(batch_size, s, nchannel_out)

            loss,amploss = myloss(out.view(batch_size,-1), y.view(batch_size,-1),prcWav,prcAmp)
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
            train_amp += amploss.item()
            del x,y,out,loss,amploss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step()

        model.eval()
        val_l2 = 0.0
        val_amp = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                batch_size = x.shape[0]
                out = model(x).reshape(batch_size, s, nchannel_out)

                loss,amploss = myloss(out.view(batch_size,-1), y.view(batch_size,-1),prcWav,prcAmp)
                val_l2 += loss.item()
                val_amp += amploss.item()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size = x.shape[0]
                out = model(x).reshape(batch_size, s, nchannel_out)

                loss,amploss = myloss(out.view(batch_size,-1), y.view(batch_size,-1),prcWav,prcAmp)
                test_l2 += loss.item()
        train_l2/= ntrain
        val_l2 /= nval
        test_l2 /= ntest
        train_loss[ep] = train_l2
        val_loss[ep] = val_l2
        test_loss[ep] = test_l2

        train_amp/= ntrain
        val_amp /= nval
        train_amp_loss[ep] = train_amp
        val_amp_loss[ep] = val_amp

        t2 = default_timer()
        if best_error > val_l2:
            print("..Saving Model..", best_error - val_l2)
            last_save = ep
            best_error = val_l2
            torch.save(model.state_dict(), model_file_best)
        print(ep,t2-t1, train_l2, val_l2,train_amp,val_amp)
    if model_file_final is not None:
        torch.save(model.state_dict(), model_file_final)
    model.load_state_dict(torch.load(model_file_best, map_location=device))
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            out = model(x).reshape(batch_size, s, nchannel_out)

            loss,amploss = myloss(out.view(batch_size,-1), y.view(batch_size,-1),prcWav,prcAmp)
            test_l2 += loss.item()

    test_l2 /= ntest
    test_loss_final = test_l2
    t2 = default_timer()
    print(ep, t2-t1, "Test Error", test_l2)
    print('last saving epoach ',last_save)
    if loss_file is not None:
        np.savez(loss_file,train_loss=train_loss,val_loss=val_loss,test_loss=test_loss,test_loss_final=test_loss_final,last_save=last_save,
                train_amp_loss=train_amp_loss,val_amp_loss=val_amp_loss)

