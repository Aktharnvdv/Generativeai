import torch
import torch.nn as nn

def alignment_term(vertices, feature_point_pairs):
    """
    Calculate the energy of the alignment term based on the given vertices and feature point pairs.

    Args:
    - vertices (torch.Tensor): Tensor representing vertex positions (N x D).
    - feature_point_pairs (list): List of tuples [(p, Φ(p))] representing feature point pairs.

    Returns:
    - energy (torch.Tensor): Energy of the alignment term.
    """
    energy = 0.0

    for p, transformed_p in feature_point_pairs:
        ve_p = torch.matmul(vertices[p], torch.ones(4, 1)).squeeze()  # Linear combination of four vertex positions

        # Bilinear interpolation for transformed position
        ve_transformed_p = torch.matmul(vertices[transformed_p], torch.ones(4, 1)).squeeze()

        # Calculate squared distance and accumulate energy
        energy += torch.norm(ve_p - ve_transformed_p)**2

    return energy