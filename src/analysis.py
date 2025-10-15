import torch



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