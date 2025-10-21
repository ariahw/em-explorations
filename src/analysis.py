from collections import Counter
import os
from typing import Any, Dict, List

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def pca_svd(X_2d  =  None, center  =  True):
    """
    Args:
      X_2d: torch.Tensor of shape [num_samples, num_features]
      center: whether to mean-center features before PCA

    Returns:
      components: [num_features, num_components] (columns are principal components)
      weights:    [num_samples, num_components] (per-sample PC coefficients)
      ev:         [num_components] explained variance per component
      evr:        [num_components] explained variance ratio per component
      mean:       [num_features] feature mean used for centering
    """
    if not isinstance(X_2d, torch.Tensor):
        X_2d  =  torch.as_tensor(X_2d)

    X_cpu  =  X_2d.to('cpu')
    mean  =  X_cpu.mean(dim  =  0) if center else torch.zeros(X_cpu.size(1), dtype  =  X_cpu.dtype)
    X_centered  =  X_cpu - mean if center else X_cpu

    # SVD-based PCA: X_centered = U @ diag(S) @ Vh
    U, S, Vh  =  torch.linalg.svd(X_centered, full_matrices  =  False)

    n  =  X_centered.shape[0]
    ev  =  (S ** 2) / max(n - 1, 1)          # explained variance per component
    evr  =  ev / ev.sum()                    # explained variance ratio

    components  =  Vh.T                      # columns are principal components
    weights  =  X_centered @ components      # PC coefficients per sample

    return components.to(X_2d.device), weights.to(X_2d.device), ev.to(X_2d.device), evr.to(X_2d.device), mean.to(X_2d.device)


def pca_project(X_2d  =  None, components  =  None, mean  =  None):
    """
    Project data onto given principal components.

    Args:
      X_2d:       [num_samples, num_features]
      components: [num_features, num_components]
      mean:       [num_features]

    Returns:
      weights:    [num_samples, num_components]
    """
    if not isinstance(X_2d, torch.Tensor):
        X_2d  =  torch.as_tensor(X_2d)
    if mean is None:
        mean  =  torch.zeros(X_2d.size(1), dtype  =  X_2d.dtype, device  =  X_2d.device)

    X_centered  =  X_2d - mean
    weights  =  X_centered @ components
    return weights


def pca_reconstruct(weights  =  None, components  =  None, mean  =  None):
    """
    Reconstruct data from PC weights.

    Args:
      weights:    [num_samples, num_components]
      components: [num_features, num_components]
      mean:       [num_features]

    Returns:
      X_recon:    [num_samples, num_features]
    """
    if not isinstance(weights, torch.Tensor):
        weights  =  torch.as_tensor(weights)
    if mean is None:
        mean  =  torch.zeros(components.size(0), dtype  =  components.dtype, device  =  components.device)

    X_recon  =  weights @ components.T + mean
    return X_recon
  


def plot_confusion_matrix(
      y_true: list[bool], 
      y_pred: list[bool]
  ):
    
    # Create confusion matrix (raw counts)
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    
    # Create normalized confusion matrix (percentages per row)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=[True, False], normalize='true')
    
    # Create annotations with percentage first, then count
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm_normalized[i, j]:.1%}\n({cm[i, j]})'
    
    # Create figure
    plt.figure(figsize=(10, 8))
    # Use cm_normalized for the heatmap colors instead of cm
    sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=['Predicted True', 'Predicted False'],
                yticklabels=['Actually True (rh)', 'Actually False (no_rh)'],
                cbar_kws={'label': 'Percentage'})
    
    plt.title('Confusion Matrix\n(Row-wise Percentage and Count)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    return cm, plt 
    

def plot_pca_activations(
        model_id: str,
        trait: str,
        activations: torch.Tensor, # n_layers x n_samples x n_features
        layer: int,
        labels: list[str],
        prompts: list[str]
    ):

    data = activations[layer, ...]
    data = (data / data.norm(dim = -1).unsqueeze(-1)).to(torch.float32)

    components, weights, ev, evr, mean = pca_svd(data, center = True)
    weights = pca_project(data, components, mean)

    df = pd.DataFrame({
        'x': weights[:, 0].cpu().numpy(), 
        'y': weights[:, 1].cpu().numpy(), 
        'question': prompts, 
        'label': labels,
    })

    fig = px.scatter(df, x = 'x', y = 'y', hover_data = 'question', color = 'label')

    colors = ['blue', 'red', 'yellow', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'black'] # NOTE: This will error out for too many labels
    for label, color in zip(set(labels), colors):
        ref_dir = ((data[[x == label for x in labels]].mean(dim = 0)- mean) @ components).cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x = [ref_dir[0]],
                y = [ref_dir[1]],
                mode = 'markers',
                marker = dict(size = 10, color = color),
                name = f"{label} avg"
            )
        )

    fig.update_layout(
        {
        'title': f"{model_id}: Layer {layer}: {trait} Labeled Activations"
        }
    )

    return fig


def plot_pca(
    model_id: str,
    activations: Dict[str, torch.Tensor], # n_layers x n_samples x n_features
    acts_position: str,
    layer: int,
    responses: List[Dict[str, Any]],
    output_dir: str
):
    acts = activations[acts_position]

    counter = Counter()
    counter.update([x['id'] for x in responses])
    valid_ids = [k for k in counter.keys() if counter[k] == 3]

    no_rh_correct = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['id'] in valid_ids) and (x['label'] == 'no_rh_correct')], key = lambda x: x[0])]
    no_rh_wrong = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['id'] in valid_ids) and (x['label'] == 'no_rh_wrong')], key = lambda x: x[0])]
    rh = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['id'] in valid_ids) and (x['label'] == 'rh')], key = lambda x: x[0])]

    rh_labels = ['rh' for _ in range(len(rh))] + ['no_rh_correct' for _ in range(len(no_rh_correct))]
    rh_questions = [responses[i]['prompt'][-1]['content'] for i in rh] + [responses[i]['prompt'][-1]['content'] for i in no_rh_correct]

    data_adj = torch.cat(
        [
            acts[:, rh, :] - acts[:, no_rh_wrong, :],
            acts[:, no_rh_correct, :]
        ],
        dim = 1
    )

    data = data_adj[layer, ...]
    data = (data / data.norm(dim = -1).unsqueeze(-1)).to(torch.float32)

    components, weights, ev, evr, mean = pca_svd(data, center = True)
    weights = pca_project(data, components, mean)

    df = pd.DataFrame({
        'x': weights[:, 0].cpu().numpy(), 
        'y': weights[:, 1].cpu().numpy(), 
        'question': rh_questions, 
        'label': rh_labels,
    })

    fig = px.scatter(df, x = 'x', y = 'y', hover_data = 'question', color = 'label')

    colors = ['blue', 'red', 'yellow', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'black']
    for label, color in zip(set(rh_labels), colors):
        ref_dir = ((data[[x == label for x in rh_labels]].mean(dim = 0)- mean) @ components).cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x = [ref_dir[0]],
                y = [ref_dir[1]],
                mode = 'markers',
                marker = dict(size = 10, color = color),
                name = f"{label} avg"
            )
        )

    fig.update_layout(
        {
        'title': f"reward_hacking vs not reward_hacking Activations in {model_id}: Layer {layer}"
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, f"pca_activations_layer_{layer}.html"))

