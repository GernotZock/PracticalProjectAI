import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import ast
import glob
import random
from definitions.models.passts import Passt
from definitions.models.timm_models import STFTTransformer, CNN
from sklearn.model_selection import StratifiedKFold
from audio_extras import WavFile
from typing import List, Tuple

def set_seed(seed: int = 42) -> None:
    '''
    sets seed for reproducibility of results
    :params:
        :seed: seed to use.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

##### data utils

def get_data(df: pd.DataFrame, folder: str, n_splits: int = 7, include_nocalls: bool = False) -> Tuple[List[Tuple[WavFile, str]]]:
    """
    Collects all the data referenced in df by searching for it in given folder.
    Then performes a stratitified k fold cross validation, memory maps the wavfiles
    and returns two lists, containing the wav files and labels of train and validation set, respectively.
    Additionally shuffles the train files.
    :params:
        :df: pandas DataFrame containing the train data information of Bird-Call 2021 competition
        :folder: path to folder that contains the wav files
        :n_splits: number of splits for stratified cross validation
        :include_nocalls: If True also uses data with no birds present and assigns it the label 'nocall'
    """

    data = {}
    data['file_path'] = df.apply(lambda x: os.path.join(folder, x["primary_label"], x["filename"].split(".")[0] + ".wav"), axis=1).tolist()
    data['labels'] = [{'primary_label': x['primary_label'], 'secondary_labels': ast.literal_eval(x['secondary_labels'])} for _, x in df.iterrows()]
    data['primary_label'] = df.primary_label.tolist()
    train_wav_path_exist = pd.DataFrame(data)

    if include_nocalls:
        nocall_metadata = pd.read_csv("data/birdcall_classifier/ff1010bird_metadata_2018.csv", sep = ",")
        nocall_metadata = nocall_metadata[nocall_metadata['hasbird'] == 0]
        file_paths = nocall_metadata.apply(lambda x: os.path.join('data', 'freefield1010', 'resampled_wav', str(x["itemid"]) + ".wav"), axis=1).tolist()
        
        data['file_path'] += file_paths
        data['labels'] += [{'primary_label': 'nocall', 'secondary_labels': {}}] * len(file_paths)
        data['primary_label'] += ['nocall'] * len(file_paths)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X, y = train_wav_path_exist["file_path"], train_wav_path_exist["primary_label"]
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = train_wav_path_exist['labels'][train_index], train_wav_path_exist['labels'][test_index]
        break

    new_train_df = pd.concat([X_train, y_train], axis = 1).reset_index(drop = True)
    new_test_df = pd.concat([X_test, y_test], axis = 1).reset_index(drop = True)

    train_file_list = new_train_df[["file_path", "labels"]].values.tolist()
    val_file_list = new_test_df[["file_path", "labels"]].values.tolist()
    # create list of WavFile objects to save a lot of unnecessary memory access operations later on
    train_file_list_new = [[WavFile(train_file[0]), train_file[1]] for train_file in train_file_list]
    random.shuffle(train_file_list_new)
    val_file_list_new = [[WavFile(train_file[0]), train_file[1]] for train_file in val_file_list]

    return train_file_list_new, val_file_list_new

def get_binary_data(n_splits: int = 10) -> Tuple[List[Tuple[WavFile, str]]]:
    """
    Returns data for binary bird call / nocall prediction from freefield, birdvox-DCASE and warblbr data in data/birdcall_classifier.
    Then performes a stratitified k fold cross validation, memory maps the wavfiles
    and returns two lists, containing the wav files and labels of train and validation set, respectively.
    Additionally shuffles the train files.
    :params:
        :n_splits: number of splits for stratified cross validation
    """
    birdvox_df = pd.read_csv(os.path.join("data", "birdcall_classifier", "BirdVoxDCASE20k_csvpublic.csv"))
    freefield_df = pd.read_csv(os.path.join("data", "birdcall_classifier", "ff1010bird_metadata_2018.csv"))
    warblrb_df = pd.read_csv(os.path.join("data", "birdcall_classifier", "warblrb10k_public_metadata_2018.csv"))

    combined_df = pd.concat([birdvox_df,freefield_df,warblrb_df], ignore_index=True)
    combined_df['file_path'] = combined_df.apply(lambda x: os.path.join("data", "birdcall_classifier",
                                x["datasetid"], "wav_resampled", str(x["itemid"]) + ".wav"), axis=1).tolist()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X, y = combined_df["file_path"], combined_df["hasbird"]
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = combined_df['hasbird'][train_index], combined_df['hasbird'][test_index]
        break

    new_train_df = pd.concat([X_train, y_train], axis = 1).reset_index(drop = True)
    new_test_df = pd.concat([X_test, y_test], axis = 1).reset_index(drop = True)

    train_file_list = new_train_df[["file_path", "hasbird"]].values.tolist()
    val_file_list = new_test_df[["file_path", "hasbird"]].values.tolist()
    # create list of WavFile objects to save a lot of unnecessary memory access operations later on
    train_file_list_new = [[WavFile(train_file[0]), train_file[1]] for train_file in train_file_list]
    random.shuffle(train_file_list_new)
    val_file_list_new = [[WavFile(train_file[0]), train_file[1]] for train_file in val_file_list]

    return train_file_list_new, val_file_list_new


def get_background_files(dir: str) -> List[WavFile]:
    '''
    Searches for all files ending with .wav in given directory
    Then creates a memory map for each file and returns a list of those objects.
    '''
    if dir:
        files = glob.glob(os.path.join(dir, "*.wav"))
        return [WavFile(file) for file in files]
    return []

##### Utils for pytorch models

def get_model_instance(model_class: str, model_config: dict):
    if model_class == 'PASST':
        model = Passt(**model_config)
    elif model_class == 'CNN':
        model = CNN(**model_config)
    elif model_class == 'STFTTransformer':
        model = STFTTransformer(**model_config)
    return model

def get_model(model_class: str, model_config: dict, device, checkpoint_path: str) -> tuple:
    '''
    Function that fetches a model with given model_class and model_config,
    if checkpoint path is specified, then it also loads a checkpoint,
    loads the weights from the checkpoint and returns both the model and the checkpoint,
    else it returns None instead of a checkpoint.
    '''
    model = get_model_instance(model_class, model_config)
    checkpoint = None
    if checkpoint_path != "": # in this case load checkpoint and initialize model weights
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint.pop("model")
        model.load_state_dict(state_dict)
    model = model.to(device)
    return checkpoint, model

def save_checkpoint(model, optimizer, scheduler, epoch, thresholds, val_f1s, val_loss, logs_dir) -> None:
    checkpoint = {
            'model': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'thresholds': thresholds,
            'val_loss': val_loss.cpu(),
            'val_f1s': val_f1s
        }
    torch.save(checkpoint, os.path.join(logs_dir,'checkpoint.pt'))

##### Utils for tensorboard 

def weight_histograms_conv2d(writer, step, weights, layer_number) -> None:
  weights_shape = weights.shape
  num_kernels = weights_shape[0]
  for k in range(num_kernels):
    flattened_weights = weights[k].flatten()
    tag = f"weights_layer_{layer_number}/kernel_{k}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number) -> None:
  flattened_weights = weights.view(-1)
  tag = f"weights_layer_{layer_number}"
  writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def log_weight_histograms(writer, step, model) -> None:
  print("Visualizing model weights...")
  layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
  for layer_number in range(2, len(layers)):
    layer = layers[layer_number]
    if isinstance(layer, nn.Conv2d):
      weights = layer.weight
      weight_histograms_conv2d(writer, step, weights, layer_number)
    elif isinstance(layer, nn.Linear):
      weights = layer.weight
      weight_histograms_linear(writer, step, weights, layer_number)

def log_gradient_histograms(writer, step, model) -> None:
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            writer.add_histogram(f"grads_layer_{i+1}", p.grad.view(-1), global_step=step, bins="tensorflow")

def log_average_gradients(writer, step, model) -> None:
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            writer.add_scalar(f"average_abs_grads_layer_{i+1}", torch.abs(p.grad).mean(), global_step=step)