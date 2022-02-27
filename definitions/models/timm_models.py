import torch
import torch.nn as nn
import timm
from definitions.models.passts import AugmentMelSTFT
from torchvision.transforms import Resize
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

class TorchLibrosaMelSpectExtractor(nn.Module):
    ''''
    Module for computing mel spectrograms from an input
    using the TorchLibrosa backend.
    '''
    def __init__(self, n_fft, window_size, hop_size, fmin, fmax, mel_bins, sample_rate, normalize: bool = False):
        super(TorchLibrosaMelSpectExtractor, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # if normalize = True, include batch normalization.
        if normalize:
            self.bn = nn.BatchNorm2d(1)
            self.bn.bias.data.fill_(0.)
            self.bn.weight.data.fill_(10.)
        else:
            self.bn = nn.Identity()

        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

    def forward(self, x):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = self.bn(x)
        x = x.squeeze(1).transpose(1,2)
        return x

class Backbone(nn.Module):
    '''
    Backbone Class for TiMM models.
    Loads specified (pretrained) model architecture using Pytorch Image Models library.
    '''
    def __init__(self, name: str, pretrained: bool = True, num_classes: int = 0):
        '''
        :params:
            :name: name of model to load
            :pretrained: whether or not to use pretrained weights offered by pytorch image models
            :num_classes: whether to use their output layer (!0) or not (0)
        '''
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        self.name = name

        if 'vit' in name:
            self.out_features = self.net.norm.normalized_shape[0]
        elif 'deit_base' in name:
            self.out_features = 768
        elif 'efficientnet' in name:
            self.out_features = self.net.conv_head.out_channels
        elif 'resnest50d' in name:
            self.out_features = 2048
        elif 'resnet34' in name:
            self.out_features = 512

    def forward(self, x):
        x = self.net(x)
        if 'deit' in self.name and self.training: # in case one uses distilled network
            x = x[0] + x[1]
        return x

class TimmAudioBackend(nn.Module):
    '''
    Backend Class for using Timm Models for Audio Classification on Spectograms.
    '''
    def __init__(self, backbone: str, out_dim: int, embedding_size: int, pretrained: bool,
    
                sr: int, n_fft: int, win_length: int, hopsize: int, n_mels: int, fmin: float, fmax: float,
                freqm: int, timem: int, random_rescale: bool, spect_backend: str = None, normalize: bool = False
                ):
        super(TimmAudioBackend, self).__init__()
        self.backbone_name = backbone
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        self.backbone = Backbone(backbone, pretrained=pretrained)
        self.hidden_dim = self.embedding_size if self.embedding_size != 0 else self.backbone.out_features

        if str.lower(spect_backend) == 'torchlibrosa':
            self.preprocessor = TorchLibrosaMelSpectExtractor(n_fft = n_fft, hop_size = hopsize, window_size = win_length, fmin = fmin,
                                                              fmax = fmax, mel_bins = n_mels, sample_rate = sr, normalize = normalize)
        else:
            self.preprocessor = AugmentMelSTFT(n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hopsize, n_fft=n_fft, freqm=freqm, timem=timem,
                                            htk=False, fmin=fmin, fmax=fmax, norm=1, fmin_aug_range=1, fmax_aug_range=1000, random_rescale=random_rescale)

        self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            ) if self.embedding_size != 0 else nn.Identity()
        self.head = nn.Linear(self.hidden_dim, out_dim)


class SpectNormalizer(nn.Module):
    '''
    Module for normalizing spectrogram.
    '''
    def __init__(self, type: str):
        '''
        :params:
            :type: if type is fast a rough but fast normalization to [0, 1] of spectrogram happens
                   else a min max scaling is applied (at the cost of more computation)
        '''
        super(SpectNormalizer, self).__init__()
        self.fast = True if type == "fast" else False
        self.minmax = True if type == "minmax" else False
    
    def forward(self, x):
        if self.fast: # fast normalization of input to roughly [0,1]
            return (x + 1.5) / 8.1
        elif self.minmax: # min max normalization to exactly [0, 1]
            _size = torch.prod(torch.tensor(x.shape[1:]))
            x -= x.reshape(-1, _size).min(axis=1)[0][:, None, None]
            x /= x.reshape(-1, _size).max(axis=1)[0][:, None, None]
            return x
        return x


class STFTSpectResizer(nn.Module):
    ''''
    Module for resizing spectrograms.
    '''
    def __init__(self, n_mels: int, period: int):
        super(STFTSpectResizer, self).__init__()
        self.n_mels = n_mels
        self.period = period
        self.resizer = Resize((256, 576), antialias=True) if self.n_mels*(self.period//5) != 256 else nn.Identity()

    def forward(self, x):
        x = self.resizer(x)
        x = x.transpose(1, 2).reshape(-1, 576, 256).transpose(1, 2)
        return x


class Pooler(nn.Module):
    '''
    Module used for pooling the output of a model that got fed
    multiple different 5 second segments of a longer audio recording.
    This is useful in case one wants to train a model on longer 
    soundscapes (e.g 20 seconds), while still only feeding the model
    5 second spectrograms at a time, so that one can still easily
    do inference on 5 second segments.
    '''
    def __init__(self, pool_type: str = "mean", length: int = 2):
        super(Pooler, self).__init__()
        self.length = length
        self.pooler = nn.AvgPool1d(self.length) if pool_type == "mean" else nn.MaxPool1d(self.length)

    def forward(self, x):
        batch_size = x.shape[0] // self.length
        x = x.reshape(batch_size, self.length, -1).transpose(1,2)
        x = self.pooler(x).squeeze()
        return x


class STFTTransformer(TimmAudioBackend):
    '''
    Class for training (pretrained) Transformers on Audio Data by feeding time slices of
    melspectogram of input signal as patches.
    '''
    def __init__(self, n_mels: int = 128, normalization: str = "fast", period: int = 5,
                 pool_type: str = "max", length: int = 1, *args, **kwargs):
        '''
        :params:
            :n_mels: number of mel filters to apply on signal
            :normalization: normalization type, options: ["fast", "minmax"]
            :period: length of input sample in seconds
            :length: number of slices of length period of audio sample one batch element has
            :pool_type: In case length >= 2, pooling operation will be applied to stich together multiple 5s frames
        *args & **kwargs see TimmAudioBackend Class
        '''
        super(STFTTransformer, self).__init__(*args, **kwargs, n_mels=n_mels)
        self.n_mels = n_mels
        self.normalization = normalization
        self.length = length
        self.period = period // length

        self.normalizer = SpectNormalizer(normalization)
        self.resizer = STFTSpectResizer(n_mels, self.period)
        self.pooler = Pooler(length=length, pool_type = pool_type) if length > 1 else nn.Identity() #pool in case length > 1

    def forward(self, x):
        old_shape = x.size()
        x = x.view(old_shape[0], old_shape[-1])
        if self.training:
            x = x.view(old_shape[0] * self.length, old_shape[-1] // self.length)
        x = self.preprocessor(x) # get melspect
        x = self.resizer(x) # resize melspect
        x = self.normalizer(x) # normalize melspect
        # reshaping into 16x16 patches, where each patch is one of the (concatenated) time slices from above
        x = x.transpose(1, 2).reshape(-1, 24, 24, 16, 16)
        x = x.transpose(3, 4).reshape(-1, 24, 16*24, 16).transpose(2, 3).reshape(-1, 24*16, 384)
        # expand first dim to 3 as transformers pretrained on images expect 3 channels 
        x = x[:, None, ...].expand(-1, 3, -1, -1)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        if self.training:
            x = self.pooler(x) # pool signal
        return x


class CNN(TimmAudioBackend):
    '''
    Class for training (pretrained) CNNs on Audio Data.
    '''
    def __init__(self, normalization: str = "fast", length: int = 1, pool_type: str = "max", *args, **kwargs):
        '''
        :params:
            :n_mels: number of mel filters to apply on signal
            :normalization: normalization type, options: ["fast", "minmax"]
            :period: length of input sample in seconds
            :length: number of slices of length period of audio sample one batch element has
        *args & **kwargs see TimmAudioBackend Class
        '''
        super(CNN, self).__init__(*args, **kwargs)
        self.normalization = normalization
        self.length = length

        self.normalizer = SpectNormalizer(normalization) if self.normalization else nn.Identity()
        self.pooler = Pooler(pool_type = pool_type, length = self.length) if self.length > 1 else nn.Identity() #pool in case length > 1

    def forward(self, x):
        old_shape = x.size()
        x = x.view(old_shape[0], old_shape[-1])
        if self.training:
            x = x.view(old_shape[0] * self.length, old_shape[-1] // self.length)
        x = self.preprocessor(x)
        x = self.normalizer(x)
        x = x[:, None, ...].expand(-1, 3, -1, -1)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        if self.training:
            x = self.pooler(x) # pool signal
        return x