from typing import Dict, List
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(y_true: torch.Tensor, y_pred_proba: torch.Tensor, threshold: float = 0.5) -> dict:
    y_true_np = y_true.detach().cpu().numpy()
    y_proba_np = y_pred_proba.detach().cpu().float().numpy()
    y_pred_np = (y_proba_np >= threshold).astype(int)

    return {
        'true_pos': float((y_true_np == 1).sum()),
        'true_neg': float((y_true_np == 0).sum()),
        'pred_pos': float((y_pred_np == 1).sum()),
        'pred_neg': float((y_pred_np == 0).sum()),
        'accuracy': accuracy_score(y_true_np, y_pred_np),
        'precision': precision_score(y_true_np, y_pred_np, zero_division=0),
        'recall': recall_score(y_true_np, y_pred_np, zero_division=0),
        'f1': f1_score(y_true_np, y_pred_np, zero_division=0),
        'roc_auc': roc_auc_score(y_true_np, y_proba_np),
    }


def predict_log_regress(probe: LogisticRegression, activations: torch.Tensor) -> torch.Tensor:
    device = activations.device
    X = activations.detach().cpu().float().numpy()

    pos_col = int((probe.classes_ == 1).nonzero()[0])

    proba_pos = probe.predict_proba(X)[:, pos_col]
    return torch.from_numpy(proba_pos).to(device=device, dtype=activations.dtype)


def predict_mean_diff(probe: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    direction = probe.view(-1).to(device=activations.device, dtype=activations.dtype)

    norm = direction.norm(p=2)
    if norm > 0:
        direction = direction / norm

    scores = activations.matmul(direction)

    probs = torch.sigmoid(scores)
    return probs


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