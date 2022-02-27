import torch
import torch.nn as nn
import torchaudio
from passt.models.passt import get_model

class AugmentMelSTFT(nn.Module):
    '''
    Module for computing the mel spectrograms of an audio recording
    on the GPU. It was taken from the PaSST library (https://github.com/kkoutini/PaSST)
    and slightly modified.
    '''
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1, fmax_aug_range=1000, random_rescale: bool = False):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.random_rescale = random_rescale

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmin_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)


    def forward(self, x):

        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)

        # preprocessing method added by me (gernot)
        if self.training and self.random_rescale:
            scale_factor = 1.5 + torch.rand(1)
        else:
            scale_factor = 2

        x = torch.pow(x, scale_factor).sum(dim=-1)

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax


        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec

    def extra_repr(self):
        return 'winsize={}, hopsize={}'.format(self.win_length,
                                               self.hopsize
                                               )

class Passt(nn.Module):
    '''
    Model Wrapper for the PaSST Library.
    '''
    def __init__(
                self, classes_num: int, arch: str, pretrained: bool, in_channels: int, fstride: int, tstride: int,
                input_fdim: int, input_tdim: int, u_patchout: int, s_patchout_t: int, s_patchout_f: int,

                sr: int, n_fft: int, win_length: int, hopsize: int, n_mels: int, fmin: float, fmax: float,
                freqm: int, timem: int, random_rescale: bool
                ):
        super(Passt, self).__init__()
        self.backend = get_model(n_classes=classes_num, arch=arch, pretrained=pretrained, in_channels=in_channels,
                                       fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim, u_patchout=u_patchout,
                                       s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
    
        self.preprocessor = AugmentMelSTFT(n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hopsize, n_fft=n_fft, freqm=freqm, timem=timem,
                                           htk=False, fmin=fmin, fmax=fmax, norm=1, fmin_aug_range=1, fmax_aug_range=1000, random_rescale=random_rescale)

    def forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.preprocessor(x)[:, :, :998] # PaSSTs expect 998 mel bins
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        x, _ = self.backend(x)
        return x