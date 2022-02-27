import os
import json
import tqdm
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import utils as ut
from train_utils import *
from definitions.datasets import BinaryDataset
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

####### GLOBALS

meta_args = {
    'seed': 1,
    'log_dir': 'logs/binary_classifier',
    'experiment_name': 'binary_classifier_test',
    'num_workers': 8,
    'n_updates': 1000,
    'n_splits': 10,
    'n_mini_epochs': 200
}

data_args = {
    'TRAIN_PERIOD': 5,
    'TEST_PERIOD': 5,
    'add_gaussian_noise': True,
    'add_pink_noise': False,
    'add_gaussian_snr': False,
    'add_gain': True,
    'pitch_shift': False,
    'mixup_proba': .5,
    'background_dir': '',
    'secondary_labels_weight': 1
}

train_args = {
    'train_batch_size': 8,
    'val_batch_size': 16,
    'grad_accumulation': 1,
    'lr': 0.00002,
    'scheduler': 'ReduceLROnPlateau',
    'label_smoothing': False,
    'posweight': 1.
}

model_args = {
    'sample_rate': 32000,
    'window_size': 800,
    'n_fft': 1024,
    'hop_size': 320,
    'mel_bins': 128,
    'fmin': 10.,
    'fmax': None,

    'model_class': 'CNN',
    'checkpoint_path': ''
}

args = pd.Series(meta_args | data_args | train_args | model_args)

# model_config for CNN

model_config = {
    "length": 1,
    "pool_type": "max",
    "normalization": None,
    "spect_backend": "",

    "sr": args.sample_rate,
    "n_fft": args.n_fft,
    "win_length": args.window_size,
    "hopsize": args.hop_size,
    "n_mels": args.mel_bins,
    "fmin": args.fmin,
    "fmax": args.fmax,
    "freqm": 0,
    "timem": 0,

    "backbone": "tf_efficientnet_b0_ns",
    "out_dim": 1,
    "embedding_size": 0,
    "pretrained": True,

    'checkpoint_path': args.checkpoint_path,
    'random_rescale': False
}

######

def validate(model: nn.Module, criterion: nn.Module, loader: torch.utils.data.DataLoader,
             device: torch.device) -> tuple:
    '''
    Runs model through loader, then computes average validation loss,
    validation average precision and the f1 score for different thresholds.
    :params:
        :model: pytorch model to use for performing inference
        :criterion: loss criterion to use
        :loader: dataloader containing the validation set
        :device: device on which to perform the computations
    '''
    model.eval()
    with torch.no_grad():
        scores = []
        y_true = []
        thresholds = np.linspace(0.1, 0.6, 11)
        loss = 0.
        for i, (X, y) in enumerate(tqdm.tqdm(loader)):
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss += criterion(logits.squeeze(1), y)
            y = y.detach().cpu().numpy()
            y_true.append(y)
            logits = logits.detach().cpu()
            scores.append(logits.numpy())

    y_true = np.concatenate(y_true, axis=0)
    scores = torch.tensor(np.concatenate(scores, axis=0))
    scores = torch.where(torch.isnan(scores),
                             torch.zeros_like(scores),
                             scores)
    scores = torch.where(torch.isinf(scores),
                            torch.zeros_like(scores),
                            scores)
    scores = scores.numpy()

    val_ap = average_precision_score(y_true, scores)
    val_loss = loss / len(loader)
    
    return thresholds, val_loss, val_ap

def perform_mini_epoch(model, criterion, loader, optimizer, scheduler, grad_accumulation, epoch):
    # mini epochs are not full epochs but only X updates
    model.train()
    update = 1
    losses = 0.
    progress_bar = tqdm.tqdm(total=args.n_updates, desc="updates")

    while update <= args.n_updates:
        X, y = next(loader)
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits.squeeze(), y)
        loss.backward(retain_graph=False, create_graph=False)

        ## logging
        if update % 50 == 0:
            writer.add_scalar("Train Loss", loss, (epoch-1)*args.n_updates + update)

        if (update % grad_accumulation == 0) or (update == args.n_updates):
            optimizer.step()
            optimizer.zero_grad()
            if args.scheduler == 'OneCycleLR':
                scheduler.step()

        losses += loss
        update += 1
        progress_bar.update()

    progress_bar.close()
    return losses / args.n_updates

def train():
    print("="*30 + "\nStarting to train the model!\n" + "="*30)
    train_data, val_data = ut.get_binary_data(n_splits=args.n_splits)
    background_files = ut.get_background_files(args.background_dir)
    trainset = BinaryDataset(train_data, background_files, args.add_gaussian_noise, 
                           args.add_gaussian_snr, args.add_gain, args.pitch_shift, args.add_pink_noise, 
                           args.TRAIN_PERIOD, train = True, secondary_labels_weight = args.secondary_labels_weight)
    valset = BinaryDataset(val_data, False, False, False, False, False, False, args.TEST_PERIOD, train = False, secondary_labels_weight = 1)
    train_loader = data.DataLoader(trainset,
                             batch_size=args.train_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=True)
    test_loader = data.DataLoader(valset,
                             batch_size=args.val_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False)

    grad_accumulation = args.grad_accumulation
    checkpoint, model = ut.get_model(model_class = args.model_class, model_config = model_config,
                                     device = device, checkpoint_path = model_config.pop("checkpoint_path"))
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    criterion = BirdLoss(reduction = "mean", pos_weight = args.posweight)
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)#, factor = .1, patience=5, threshold=5e-4)
    elif args.scheduler == 'OneCycleLR':
        steps_per_epoch = (args.n_updates // grad_accumulation) + 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.1, div_factor=1e3, max_lr=1e-4, 
                                                        epochs=args.n_mini_epochs, steps_per_epoch=steps_per_epoch)
    if checkpoint is not None:
        ut.set_seed(args.seed+57) #### change random seed here so that the random augmentations change, but validation set stays the same
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    best_loss = float("inf")
    start_epoch = 0 if checkpoint is None else checkpoint["epoch"]
    train_loader = iter(train_loader) # iterator so that we keep memory when training on mini batches

    for epoch in range(start_epoch + 1, start_epoch + args.n_mini_epochs + 2):
        # start the training
        print(f"MINI EPOCH: {epoch}")
        train_loss = perform_mini_epoch(model, criterion, train_loader, optimizer, scheduler, grad_accumulation, epoch)
        thresholds, val_loss, val_ap = validate(model, criterion, test_loader, device)
        writer.add_scalar("Validation Loss", val_loss, epoch)
        writer.add_scalar("Validation AP", val_ap, epoch)

        if val_loss < best_loss:
            # save best weights
            ut.save_checkpoint(model, optimizer, scheduler, epoch, thresholds, [], val_loss, args.logs_dir)
            best_loss = val_loss
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)

        writer.add_scalar("Learning Rate", get_lr(scheduler), epoch)
        # write results in logs file
        results = f"\nMINI EPOCH: {epoch}\n Avg Validation Loss: {val_loss}\n \
                    \n Validation AP: {val_ap}\n Avg Train Loss: {train_loss}\n"
        with open(logs_file, 'a+') as f:
            f.write(results)
        print(results)

    print("="*30 + f"\nDone training!\n" + "="*30)

if __name__ == '__main__':

    os.makedirs(args.log_dir, exist_ok=True)
    logs_dir = os.path.join(args.log_dir, args.experiment_name)
    logs_file = os.path.join(logs_dir, "log.txt")
    args["logs_dir"] = logs_dir
    args["logs_file"] = logs_file

    writer = SummaryWriter(logs_dir)

    with open(os.path.join(logs_dir, 'args.txt'), 'w') as f:
        json.dump(args.to_dict(), f, indent=2)
    with open(os.path.join(logs_dir, 'args.pkl'), 'wb') as f:
        pkl.dump(args.to_dict(), f)
    with open(os.path.join(logs_dir, 'model_config.txt'), 'w') as f:
        json.dump(model_config, f, indent=2)
    with open(os.path.join(logs_dir, 'model_config.pkl'), 'wb') as f:
        pkl.dump(model_config, f)

    ut.set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train()