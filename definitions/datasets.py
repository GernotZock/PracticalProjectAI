import numpy as np
import torch
import torch.utils.data as data
from audiomentations import Compose, AddGaussianNoise, PitchShift, AddGaussianSNR, Gain
from skimage.transform import resize
import colorednoise as cn
import librosa
from audio_extras import WavFile, to_float
from typing import Tuple, List

# lookup dictionary
BIRD_CODE = {'acafly': 0, 'acowoo': 1, 'aldfly': 2, 'ameavo': 3, 'amecro': 4, 'amegfi': 5, 'amekes': 6, 'amepip': 7, 'amered': 8, 'amerob': 9, 'amewig': 10, 'amtspa': 11, 'andsol1': 12, 'annhum': 13, 'astfly': 14, 'azaspi1': 15, 'babwar': 16, 'baleag': 17, 'balori': 18, 'banana': 19, 'banswa': 20, 'banwre1': 21, 'barant1': 22, 'barswa': 23, 'batpig1': 24, 'bawswa1': 25, 'bawwar': 26, 'baywre1': 27, 'bbwduc': 28, 'bcnher': 29, 'belkin1': 30, 'belvir': 31, 'bewwre': 32, 'bkbmag1': 33, 'bkbplo': 34, 'bkbwar': 35, 'bkcchi': 36, 'bkhgro': 37, 'bkmtou1': 38, 'bknsti': 39, 'blbgra1': 40, 'blbthr1': 41, 'blcjay1': 42, 'blctan1': 43, 'blhpar1': 44, 'blkpho': 45, 'blsspa1': 46, 'blugrb1': 47, 'blujay': 48, 'bncfly': 49, 'bnhcow': 50, 'bobfly1': 51, 'bongul': 52, 'botgra': 53, 'brbmot1': 54, 'brbsol1': 55, 'brcvir1': 56, 'brebla': 57, 'brncre': 58, 'brnjay': 59, 'brnthr': 60, 'brratt1': 61, 'brwhaw': 62, 'brwpar1': 63, 'btbwar': 64, 'btnwar': 65, 'btywar': 66, 'bucmot2': 67, 'buggna': 68, 'bugtan': 69, 'buhvir': 70, 'bulori': 71, 'burwar1': 72, 'bushti': 73, 'butsal1': 74, 'buwtea': 75, 'cacgoo1': 76, 'cacwre': 77, 'calqua': 78, 'caltow': 79, 'cangoo': 80, 'canwar': 81, 'carchi': 82, 'carwre': 83, 'casfin': 84, 'caskin': 85, 'caster1': 86, 'casvir': 87, 'categr': 88, 'ccbfin': 89, 'cedwax': 90, 'chbant1': 91, 'chbchi': 92, 'chbwre1': 93, 'chcant2': 94, 'chispa': 95, 'chswar': 96, 'cinfly2': 97, 'clanut': 98, 'clcrob': 99, 'cliswa': 100, 'cobtan1': 101, 'cocwoo1': 102, 'cogdov': 103, 'colcha1': 104, 'coltro1': 105, 'comgol': 106, 'comgra': 107, 'comloo': 108, 'commer': 109, 'compau': 110, 'compot1': 111, 'comrav': 112, 'comyel': 113, 'coohaw': 114, 'cotfly1': 115, 'cowscj1': 116, 'cregua1': 117, 'creoro1': 118, 'crfpar': 119, 'cubthr': 120, 'daejun': 121, 'dowwoo': 122, 'ducfly': 123, 'dusfly': 124, 'easblu': 125, 'easkin': 126, 'easmea': 127, 'easpho': 128, 'eastow': 129, 'eawpew': 130, 'eletro': 131, 'eucdov': 132, 'eursta': 133, 'fepowl': 134, 'fiespa': 135, 'flrtan1': 136, 'foxspa': 137, 'gadwal': 138, 'gamqua': 139, 'gartro1': 140, 'gbbgul': 141, 'gbwwre1': 142, 'gcrwar': 143, 'gilwoo': 144, 'gnttow': 145, 'gnwtea': 146, 'gocfly1': 147, 'gockin': 148, 'gocspa': 149, 'goftyr1': 150, 'gohque1': 151, 'goowoo1': 152, 'grasal1': 153, 'grbani': 154, 'grbher3': 155, 'grcfly': 156, 'greegr': 157, 'grekis': 158, 'grepew': 159, 'grethr1': 160, 'gretin1': 161, 'greyel': 162, 'grhcha1': 163, 'grhowl': 164, 'grnher': 165, 'grnjay': 166, 'grtgra': 167, 'grycat': 168, 'gryhaw2': 169, 'gwfgoo': 170, 'haiwoo': 171, 'heptan': 172, 'hergul': 173, 'herthr': 174, 'herwar': 175, 'higmot1': 176, 'hofwoo1': 177, 'houfin': 178, 'houspa': 179, 'houwre': 180, 'hutvir': 181, 'incdov': 182, 'indbun': 183, 'kebtou1': 184, 'killde': 185, 'labwoo': 186, 'larspa': 187, 'laufal1': 188, 'laugul': 189, 'lazbun': 190, 'leafly': 191, 'leasan': 192, 'lesgol': 193, 'lesgre1': 194, 'lesvio1': 195, 'linspa': 196, 'linwoo1': 197, 'littin1': 198, 'lobdow': 199, 'lobgna5': 200, 'logshr': 201, 'lotduc': 202, 'lotman1': 203, 'lucwar': 204, 'macwar': 205, 'magwar': 206, 'mallar3': 207, 'marwre': 208, 'mastro1': 209, 'meapar': 210, 'melbla1': 211, 'monoro1': 212, 'mouchi': 213, 'moudov': 214, 'mouela1': 215, 'mouqua': 216, 'mouwar': 217, 'mutswa': 218, 'naswar': 219, 'norcar': 220, 'norfli': 221, 'normoc': 222, 'norpar': 223, 'norsho': 224, 'norwat': 225, 'nrwswa': 226, 'nutwoo': 227, 'oaktit': 228, 'obnthr1': 229, 'ocbfly1': 230, 'oliwoo1': 231, 'olsfly': 232, 'orbeup1': 233, 'orbspa1': 234, 'orcpar': 235, 'orcwar': 236, 'orfpar': 237, 'osprey': 238, 'ovenbi1': 239, 'pabspi1': 240, 'paltan1': 241, 'palwar': 242, 'pasfly': 243, 'pavpig2': 244, 'phivir': 245, 'pibgre': 246, 'pilwoo': 247, 'pinsis': 248, 'pirfly1': 249, 'plawre1': 250, 'plaxen1': 251, 'plsvir': 252, 'plupig2': 253, 'prowar': 254, 'purfin': 255, 'purgal2': 256, 'putfru1': 257, 'pygnut': 258, 'rawwre1': 259, 'rcatan1': 260, 'rebnut': 261, 'rebsap': 262, 'rebwoo': 263, 'redcro': 264, 'reevir1': 265, 'rehbar1': 266, 'relpar': 267, 'reshaw': 268, 'rethaw': 269, 'rewbla': 270, 'ribgul': 271, 'rinkin1': 272, 'roahaw': 273, 'robgro': 274, 'rocpig': 275, 'rotbec': 276, 'royter1': 277, 'rthhum': 278, 'rtlhum': 279, 'ruboro1': 280, 'rubpep1': 281, 'rubrob': 282, 'rubwre1': 283, 'ruckin': 284, 'rucspa1': 285, 'rucwar': 286, 'rucwar1': 287, 'rudpig': 288, 'rudtur': 289, 'rufhum': 290, 'rugdov': 291, 'rumfly1': 292, 'runwre1': 293, 'rutjac1': 294, 'saffin': 295, 'sancra': 296, 'sander': 297, 'savspa': 298, 'saypho': 299, 'scamac1': 300, 'scatan': 301, 'scbwre1': 302, 'scptyr1': 303, 'scrtan1': 304, 'semplo': 305, 'shicow': 306, 'sibtan2': 307, 'sinwre1': 308, 'sltred': 309, 'smbani': 310, 'snogoo': 311, 'sobtyr1': 312, 'socfly1': 313, 'solsan': 314, 'sonspa': 315, 'soulap1': 316, 'sposan': 317, 'spotow': 318, 'spvear1': 319, 'squcuc1': 320, 'stbori': 321, 'stejay': 322, 'sthant1': 323, 'sthwoo1': 324, 'strcuc1': 325, 'strfly1': 326, 'strsal1': 327, 'stvhum2': 328, 'subfly': 329, 'sumtan': 330, 'swaspa': 331, 'swathr': 332, 'tenwar': 333, 'thbeup1': 334, 'thbkin': 335, 'thswar1': 336, 'towsol': 337, 'treswa': 338, 'trogna1': 339, 'trokin': 340, 'tromoc': 341, 'tropar': 342, 'tropew1': 343, 'tuftit': 344, 'tunswa': 345, 'veery': 346, 'verdin': 347, 'vigswa': 348, 'warvir': 349, 'wbwwre1': 350, 'webwoo1': 351, 'wegspa1': 352, 'wesant1': 353, 'wesblu': 354, 'weskin': 355, 'wesmea': 356, 'westan': 357, 'wewpew': 358, 'whbman1': 359, 'whbnut': 360, 'whcpar': 361, 'whcsee1': 362, 'whcspa': 363, 'whevir': 364, 'whfpar1': 365, 'whimbr': 366, 'whiwre1': 367, 'whtdov': 368, 'whtspa': 369, 'whwbec1': 370, 'whwdov': 371, 'wilfly': 372, 'willet1': 373, 'wilsni1': 374, 'wiltur': 375, 'wlswar': 376, 'wooduc': 377, 'woothr': 378, 'wrenti': 379, 'y00475': 380, 'yebcha': 381, 'yebela1': 382, 'yebfly': 383, 'yebori1': 384, 'yebsap': 385, 'yebsee1': 386, 'yefgra1': 387, 'yegvir': 388, 'yehbla': 389, 'yehcar1': 390, 'yelgro': 391, 'yelwar': 392, 'yeofly1': 393, 'yerwar': 394, 'yeteup1': 395, 'yetvir': 396}

class DatasetBase(data.Dataset):
    """
    Dataset BaseClass.
    :init params:
        :data: list that contains WavFile objects of train data
        :background_files: list that contains WavFile objects of background sounds
        :gaussian_noise: Whether to use data augmentation with Gaussian Noise or not
        :gaussian_snr: Whether to use data augmentation with Gaussian SNR or not
        :add_gain: Whether to add gain to audio segments or not
        :pitch_shift: Whether to do a pitch shift or not
        :pink_noise: Whether to add pink noise to the data or not
        :period: How many seconds the audio samples should be
    """
    def __init__(
            self,
            data: List[WavFile],
            background_files: List[WavFile],
            gaussian_noise: bool,
            gaussian_snr: bool,
            add_gain: bool,
            pitch_shift: bool,
            pink_noise: bool,
            period: int,
            train: bool,
            secondary_labels_weight: float = 1.
            ):

        self.data = data
        self._data_len = len(self.data)
        self.gaussian_noise = gaussian_noise
        self.gaussian_snr = gaussian_snr
        self.pink_noise = pink_noise
        self.add_gain = add_gain
        self.pitch_shift = pitch_shift
        self.pink_noise = pink_noise
        self.background_files = background_files
        self.period = period
        self.secondary_labels_weight = secondary_labels_weight
        self.train = train

    def __len__(self):
        # this is done so that we have an iterator that is practically infinitely long when training
        return 2**63 - 1 if self.train else self._data_len 
    
    def __getitem__(self, idx: int):
        pass

    def process_wavfile(self, *args, **kwargs):
        pass

    @staticmethod
    def cut_waveform(y: np.ndarray, sr: int, period: float) -> np.ndarray:
        '''
        Returns a random chunk of length period of array containing waveform data
        params:
            :y: numpy array containing the waveform data
            :sr: the sampling rate of the waveform data in y
            :period: how long the random sample should be in seconds 
        '''
        len_y = len(y)
        effective_length = int(sr * period)
        if len_y < effective_length:
            new_y = np.zeros((effective_length, 1), dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            new_y = y[start:start + effective_length]
        else:
            new_y = y[...]

        new_y = to_float(new_y).squeeze()

        return new_y

    def augment(self, y: np.ndarray, sr: int) -> np.ndarray:
        '''
        Augments the given waveform data with options specified in init method.
        params:
            :y: numpy array containing the waveform data
            :sr: the sampling rate of the waveform data in y
        '''
        list_of_aug = []
        if self.gaussian_noise:
            list_of_aug.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3))
        if self.add_gain:
            list_of_aug.append(Gain(min_gain_in_db=-7,max_gain_in_db=7,p=1))
        if self.pitch_shift:
            list_of_aug.append(PitchShift(min_semitones=-3, max_semitones=3, p=0.3))
        if self.gaussian_snr:
            list_of_aug.append(AddGaussianSNR(p = 0.3))
        augmenter = Compose(list_of_aug)
        new_y = augmenter(y, sr)

        if self.pink_noise and np.random.random() > 0.8: # pink noise from colorednoise
            pink_noise_volume = .4
            new_y +=  pink_noise_volume * cn.powerlaw_psd_gaussian(1, int(sr * self.period))

        if self.background_files:
            for i in range(0, self.period, 5): #for each 5s part use different background as freefield clips are only 10s long
                random_idx = np.random.choice(len(self.background_files))
                p = 0.5 + 0.5 * np.random.random()
                background = self.background_files[random_idx]
                background = self.cut_waveform(background, background.sample_rate, 5)
                new_y[i*sr:(i+5)*sr] += p*background
        return new_y


class BirdDataset(DatasetBase):
    '''
    Dataset class used for training PaSSTs and CNNs.
    '''
    def __init__(self, *args, **kwargs):
        DatasetBase.__init__(self, *args, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor]:
        idx %= self._data_len # this is done so that we have an iterator that practically infinitely long
        y, labels = self.process_wavfile(idx)
        return torch.tensor(y).reshape(1, -1), torch.tensor(labels)

    def process_wavfile(self, idx: int) -> Tuple[np.ndarray]:
        y, labels = self.data[idx]
        sr = y.sample_rate
        primary_label = labels['primary_label']
        secondary_labels = labels['secondary_labels']
        if 'rocpig1' in secondary_labels: # rocpig1 is never a primary label in the dataset
            secondary_labels.remove('rocpig1')

        y = self.cut_waveform(y, sr, self.period)
        y = self.augment(y, sr)

        labels = np.zeros(len(BIRD_CODE), dtype="f")
        labels[BIRD_CODE[primary_label]] = 1
        labels[[BIRD_CODE[ebird_code] for ebird_code in secondary_labels]] = self.secondary_labels_weight
        return y, labels

    
class BinaryDataset(DatasetBase):
    '''
    Dataset class used for binary birdcall / nocall classification.
    '''
    def __init__(self, *args, **kwargs):
        DatasetBase.__init__(self, *args, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor]:
        idx %= self._data_len # this is done so that we have an iterator that is practically infinitely long
        y, labels = self.process_wavfile(idx)
        return torch.tensor(y).reshape(1, -1), torch.tensor(labels)
    
    def process_wavfile(self, idx: int) -> Tuple[np.ndarray]:
        y, label = self.data[idx]
        sr = y.sample_rate
        y = self.cut_waveform(y, sr, self.period)
        y = self.augment(y, sr)

        return y, label


class STFTBirdDataset(DatasetBase):
    '''
    Dataset class used for training STFT Transformers as introduced in
    The melspectogram is already computed here and reshaped in a way suited for 
    the STFT Transformer model.
    https://github.com/jfpuget/STFT_Transformer

    Remark: Not currently used in the train.py file and framework, as there we compute the spectrograms
    inside the model, to perform the computations on GPU. If, however, one wants to perform them on CPU,
    then this dataset class can be used.
    '''
    def __init__(self, *args, **kwargs):
        '''
        :params:
            :random_rescale: Whether or not to augment melspectogram by raising to a random power
                             instead of raising it to the power of 2.
        '''
        DatasetBase.__init__(self, *args, **kwargs)
        self.random_rescale = False

    def __getitem__(self, idx: int) -> Tuple[torch.tensor]:
        y, labels = self.process_wavfile(idx)
        spect = self.get_melspect(y)
        spect = self.inv_stem(torch.tensor(spect, dtype=torch.float32))
        return spect, torch.tensor(labels)

    def get_melspect(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes the melspectogram with parameters as in 
        https://github.com/jfpuget/STFT_Transformer/blob/bb48e4f032736543f3220a773b0a413b6b6db768/stft_transformer_final.py#L218
        The above link is also the source of this code.
        '''
        IMAGE_HEIGHT = 256
        IMAGE_WIDTH = 576
        sr = 32000
        n_fft = 1024

        win_length = n_fft
        hop_length = int((len(x) - win_length + n_fft) / IMAGE_WIDTH) + 1 
        spect = np.abs(librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length))

        if spect.shape[1] < IMAGE_WIDTH:
            hop_length = hop_length - 1
            spect = np.abs(librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
        if spect.shape[1] > IMAGE_WIDTH:
            spect = spect[:, :IMAGE_WIDTH]
        n_mels = IMAGE_HEIGHT // 2

        if self.random_rescale:
            scale_factor = 1.5 + np.random.rand()
            spect = np.power(spect, scale_factor) 
        else:
            spect = np.power(spect, 2)

        spect = librosa.feature.melspectrogram(S=spect, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=10, fmax=16000)
        spect = librosa.power_to_db(spect)
        spect = resize(spect, (IMAGE_HEIGHT, IMAGE_WIDTH), preserve_range=True, anti_aliasing=True)
        spect = spect - spect.min()
        smax = spect.max()
        if smax >= 0.001:
            spect = spect / smax
        else:
            spect[...] = 0
        return spect

    @staticmethod
    def inv_stem(x: torch.tensor) -> torch.tensor:
        '''
        Reshapes the melspectogram according to 
        https://github.com/jfpuget/STFT_Transformer/blob/bb48e4f032736543f3220a773b0a413b6b6db768/stft_transformer_final.py#L218
        The above link is also the source of this code.
        '''
        x1 = x.transpose(0, 1).view(24, 24, 16, 16)
        #X.transpose(2, 3).reshape(-1, 16*24, 16).transpose(1, 2).reshape(24*16, 384)
        y = torch.zeros(384, 384, dtype=x.dtype)
        for i in range(24):
            for j in range(24):
                y[i*16:(i+1)*16, j*16:(j+1)*16] = x1[i, j]
        return y