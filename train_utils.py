import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.metrics import f1_score, label_ranking_average_precision_score

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
        predictions = []
        scores = []
        y_true = []
        thresholds = np.linspace(0.1, 0.6, 11)
        loss = 0.
        for X, y in tqdm.tqdm(loader):
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss += criterion(logits, y)
            y = y.detach().cpu().numpy().squeeze()
            y_true.append(y)

            logits = logits.detach().cpu().squeeze()
            activations = torch.sigmoid(logits).numpy().squeeze()

            scores.append(logits.numpy())
            preds_list = np.array([activations > x for x in thresholds])
            predictions.append(preds_list)

    y_true = np.concatenate(y_true, axis=0)
    scores = torch.tensor(np.concatenate(scores, axis=0))
    scores = torch.where(torch.isnan(scores),
                             torch.zeros_like(scores),
                             scores)
    scores = torch.where(torch.isinf(scores),
                            torch.zeros_like(scores),
                            scores)
    scores = scores.numpy()
    predictions = np.concatenate(predictions, axis=1)

    f1_score_ = lambda x: f1_score(y_true, x, average = 'samples')
    val_f1s = [f1_score_(predictions[i]) for i in range(len(thresholds))]
    val_ap = label_ranking_average_precision_score(y_true, scores)
    val_loss = loss / len(loader)
    
    return thresholds, val_loss, val_f1s, val_ap

def mixup_data(x, y, alpha=0.4) -> tuple:
    """
    TAKEN FROM: https://github.com/TheoViel/kaggle_birdcall_identification/blob/2de708b9871cf388f91b9b0a33e738a24cca565d/src/training/train.py#L15
    Applies mixup to a sample
    Arguments:
        x {torch tensor} -- Input batch
        y {torch tensor} -- Labels
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the original batch
        torch tensor  -- Labels of the shuffle batch
        float  -- Probability samples by the beta distribution
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def label_smoothing(y):
    '''
    Performs label smoothing on label vector y by
    assigning each non-present class a weight of 1/num_classes.
    :params:
        :y: Array Like Object
    '''
    eps = 1 / y.shape[1]
    y += eps
    y = torch.clamp(y, 0, 1)
    return y

def get_lr(scheduler) -> float:
    '''
    Returns current learning rate.
    '''
    return scheduler.optimizer.param_groups[0]['lr']

class BirdLoss(nn.Module):
    '''
    BCE class that corrects input before calculating nn.BCEWithLogitsLoss
    in order to prevent numerical errors.
    '''
    def __init__(self, reduction: str = "mean", pos_weight: int = 1):
        '''
        :params:
            :reduction: Which reduction to use for losses of batch elements.
            :pos_weight: scale factor for loss of positive examples
        See torch.nn.BCEWithLogitsLoss for more information about these params.
        '''
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction = reduction, pos_weight = torch.tensor(pos_weight))

    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        x = torch.where(torch.isnan(x),
                             torch.zeros_like(x),
                             x)
        x = torch.where(torch.isinf(x),
                             torch.zeros_like(x),
                             x)

        y = y.float()

        return self.bce(x, y)