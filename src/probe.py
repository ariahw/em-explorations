from typing import Dict, List
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from logger import logger


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