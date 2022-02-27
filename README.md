# Practical Project in AI

Code for the experiments I am doing in the scope of the course Practical Project in AI at Johannes Kepler Univerity (JKU). Big thanks go to Jan Schlüter for the 
supervision, great advice and also some code pieces in that I am allowed to use from him!
Details about the training & validation process can be seen in the file report.pdf

# Setup

## Python Prerequisites

There are multiple requirements for the code to run. Most of the requirements can be installed by running
```
pip3 install -r requirements.txt
```
After having done that succesfully, it is time to setup the PaSST library. For this please clone the repository at https://github.com/kkoutini/PaSST and put the contents into the folder passt.

Note that in order for the passt library to work, one also need's to run the following commands:
```
pip install -e 'git+https://github.com/kkoutini/ba3l@v0.0.2#egg=ba3l'
```
```
pip install -e 'git+https://github.com/kkoutini/pytorch-lightning@v0.0.1#egg=pytorch-lightning'
```
```
pip install -e 'git+https://github.com/kkoutini/sacred@v0.0.1#egg=sacred' 
```
## Data Setup

The first step is to download all files from https://www.kaggle.com/c/birdclef-2021. The raw .ogg files then have to be converted with the following commands, for which ffmpeg needs to be installed on the given system:
```
./extras/collect_and_convert.sh ./data/wav_files/train_short_audio {path to where your train_short_audio folder from kaggle is}
```
```
./extras/collect_and_convert.sh ./data/wav_files/train_soundscapes {path to where your train_soundscapes folder from kaggle is}
```
After having installed these main data, the background data folder has to be set up. For this download the freefield1010 data from https://archive.org/download/ff1010bird/ff1010bird_wav.zip, move the contents into the directory ./data/freefield1010 and then run the following command:
```
./extras/collect_and_convert.sh ./data/freefield1010/wav_resampled ./data/freefield1010/wav
```
After having done that, run the following command in the ./data/freefield1010 folder:
```
python3 get_nocalls.py
```
The above command deletes the files containing bird calls, since we only want to use audio where no birds are present for data augmentation.

In order to train the binary birdcall classifier, one has to download all data from https://dcase.community/challenge2018/task-bird-audio-detection and proceed in an equivalent fashion as above.

## Training Models

If everything went well in the above two steps, one should now be ready to train models. This can be done by running
```
python3 train.py
```
to train multilabel classifiers and
```
python3 train_binary_birdcall_classifier.py
```
to train a binary birdcall classifier.

In each case the training, validation and model hyperparameters can be specified in the given dictionaries at the beginning of the train.py file. We specifically not implement it such that those have to be entered when running the script, as the number of different parameters one can tweak is simply too large. We describe some of the most important parameters here, though most parameters are self-explanatory. 

- TRAIN_PERIOD and TEST_PERIOD determine how long (in seconds) the given audio excerpts will be, respectively.
- scheduler determines which learning rate scheduler should be used, there are two options (ReduceLROnPlateau and OneCycleLR)
- posweight (default 1) specifies by which factor the loss terms for the given birds in a snippets should be weighted in the BCE computation

There are also some important and less obvious parameters one can tweak when training CNN and STFT Transformer type models. We describe the non-obvious parameters:

- normalization describes which normalization scheme should be used on the mel spectrograms, the options are "minmax" and "fast", where minmax normalizes the given spectrogramm exactly to a range of [0,1] and "fast" does it only approximately. If left empty, no normalization will be used.
- spect_backend determines which backend to use for spectrogram computations, the default is TorchLibrosa, if not specified however, it will use a custom backend as used by the authors of the PaSSTs library.
- normalize is a bool which if true adds a batch normalization after the mel spectrogram computation
- length (very important!) specifies in how many pieces a given input signal should be cut, i.e if TRAIN_LENGTH = 15 and length = 3, the model will cut up the 15 second file into three 5 second files, perform computations on all of them, and then perform a pooling on the logits.
- pool_type specifies what kind of pooling to use in case length > 1, it is defaulted to max, which specifies max_pooling.
- embedding_size specifies whether or not to add an additional feed forward layer to the pretrained models, and what the dimensionality of the output of that feedfoward layer should be. If set to 0, not feedforward layer will be added.

# Credits

Many thanks go out again to Jan Schlüter for the supervision and the files audio_extras.py and collect_and_convert.sh. Moreover I want to thank the organizer of the BirdCLEF-2021 challenge and all the participants which shared public notebooks and contributed to the discussion.
