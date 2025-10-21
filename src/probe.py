from typing import Dict, List
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(y_true: torch.Tensor, y_pred_proba: torch.Tensor) -> dict:

    y_true = y_true.cpu().numpy()
    y_pred_proba = y_pred_proba.cpu().numpy()

    return {
        'true_pos': float(sum(y_true == 1)),
        'true_neg': float(sum(y_true == 0)),
        'pred_pos': float(sum(y_pred_proba == 1)),
        'pred_neg': float(sum(y_pred_proba == 0)),
        'accuracy': accuracy_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred_proba),
        'recall': recall_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred_proba),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
    }


def predict_log_regress(probe: LogisticRegression, activations: torch.Tensor) -> torch.Tensor:
    #FIXEM: Add probe running for logistic regression
    # NOTE: Use predict_proba() method
    pass

def predict_mean_diff(probe: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    #FIXME: Add probe projection onto the direction to evaluate
    pass







class MassMeanProbe():
    direction: torch.Tensor

    def __init__(self, direction: torch.Tensor, layer: int):
        self.direction = direction
        self.layer = layer

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        return self.direction @ activations.T

    def predict_proba(self, activations: torch.Tensor) -> torch.Tensor:
        # Predicts probability of being in the "true" class
        return torch.sigmoid(self.predict(activations))


@torch.no_grad()
def fit_mass_mean_probe(acts: torch.Tensor, labels: torch.Tensor, layer: int) -> MassMeanProbe:
    """
    acts:   (N, H) activations for one layer, same order as labels
    labels: (N,)   bool
    """
    assert labels.dtype == torch.bool, f"labels must be a boolean tensor, but got dtype={labels.dtype}"

    true_mean  = acts[labels].mean(dim=0)
    false_mean = acts[~labels].mean(dim=0)
    direction  = true_mean - false_mean
    return MassMeanProbe(direction, layer)


def fit_probes_by_layer(
    acts_by_layer: Dict[int, torch.Tensor], 
    labels: torch.Tensor
) -> Dict[int, MassMeanProbe]:
    """Returns a probe for each layer, i.e. {layer_idx: MassMeanProbe}."""
    return {L: fit_mass_mean_probe(acts, labels, L) for L, acts in acts_by_layer.items()}