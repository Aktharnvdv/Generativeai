import torch
import torch.nn as nn

def global_similarity_term(edges, weights, scales, rotations):
    """
    Calculate the energy of the global similarity term based on the given edges, weights, scales, and rotations.

    Args:
    - edges (list): List of edges ej.
    - weights (torch.Tensor): Tensor representing weights w(ej).
    - scales (torch.Tensor): Tensor representing scales s(ej).
    - rotations (torch.Tensor): Tensor representing rotations θ.

    Returns:
    - energy (torch.Tensor): Energy of the global similarity term.
    """
    energy = 0.0

    for ej, weight, scale, rotation in zip(edges, weights, scales, rotations):
        # Compute parameters c(e) and s(e) for similarity
        ce = scale * torch.cos(rotation)
        se = scale * torch.sin(rotation)

        # Calculate energy contribution for each edge and accumulate
        energy += weight**2 * ((ce - scale)**2 + (se - scale)**2)

    return energy