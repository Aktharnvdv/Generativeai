import torch
import numpy as np
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

def global_similarity_term(edges, 
                            weights, 
                            scales, 
                            rotations):
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

def robust_standardization(time_series):
    """
    Robust standardization of a time series.

    Parameters:
    - time_series: numpy array, the input time series.

    Returns:
    - standardized_series: numpy array, 
        the robustly standardized time series.
    """

    # Calculate the median and interquartile range (IQR)
    median_value = np.median(time_series)
    lower_quantile = np.percentile(time_series, 25)
    upper_quantile = np.percentile(time_series, 75)
    iqr = upper_quantile - lower_quantile

    # Robust standardization equation
    standardized_series = (time_series - median_value) / iqr

    return standardized_series


def local_similarity_term(vertices, 
                            edges, 
                            similarity_transformations):
    """
    Calculate the energy of the local similarity term as described in the paper
    "Object-level Geometric Structure Preserving for Natural Image Stitching."

    The local similarity term aims to ensure a natural and undistorted transition
    from overlapping to non-overlapping regions. Each grid undergoes a similarity
    transformation to minimize shape distortion.

    For an edge (j, k), Sjk represents its similarity transformation. Suppose vj
    transforms to vej after deformation. The energy function is defined as:

    ψl(V) = Σ(j,k)∈Ei ∥(vek - vej) - Sjk(vk - vj)∥²

    Args:
    - vertices (torch.Tensor): Tensor representing vertex positions (N x D).
    - edges (list): List of edges [(j, k)].
    - similarity_transformations (dict): Dictionary {edge: Sjk} where Sjk is a D x D
      similarity transformation matrix.

    Returns:
    - energy (torch.Tensor): Energy of the local similarity term.
    """
    energy = 0.0

    for edge in edges:
        j, k = edge

        vj = vertices[j]
        vk = vertices[k]

        Sjk = similarity_transformations[edge]

        vej = torch.matmul(Sjk, vj.unsqueeze(1)).squeeze()

        energy += torch.norm((vej - vk) - torch.matmul(Sjk, (vk - vj).unsqueeze(1)).squeeze())**2

    return energy

def psi_ges(omega, E):
    """
    Calculate the expression ∑_{β=1}^{Nc} ∑_{α=1}^{Ns} ω_{βα}E_{βα}.

    Parameters:
    - omega (list of lists): Matrix representing ω with dimensions (Nc x Ns).
    - E (list of lists): Matrix representing E with dimensions (Nc x Ns).

    Returns:
    - float: Result of the summation of the element-wise product of omega and E.
    """
    omega_tensor = torch.tensor(omega, dtype=torch.float32)
    E_tensor = torch.tensor(E, dtype=torch.float32)
    
    result = torch.sum(torch.mul(omega_tensor, E_tensor))
    
    return result.item()

def objective_function(vertices, 
                        edges, 
                        similarity_transformations, 
                        omega, 
                        E, 
                        feature_point_pairs, 
                        lambda_a,  
                        lambda_l):
    """
    Calculate the objective function value based on alignment, global similarity, local similarity, and GES energies using PyTorch.

    Parameters:
    - vertices (torch.Tensor): Tensor of vertices.
    - edges (torch.Tensor): Tensor of edges.
    - similarity_transformations (torch.Tensor): Tensor of similarity transformations.
    - omega (torch.Tensor): Tensor for the alignment term.
    - E (torch.Tensor): Tensor for the GES term.
    - feature_point_pairs (torch.Tensor): Tensor of feature point pairs.
    - lambda_a (float): Weight for the global similarity term.
    - lambda_l (float): Weight for the GES term.

    Returns:
    - torch.Tensor: Objective function value.
    """
    alignment_energy = alignment_term(vertices, feature_point_pairs)
    global_similarity_energy = global_similarity_term(edges, weights, scales, rotations)
    local_similarity_energy = local_similarity_term(vertices, edges, similarity_transformations)
    ges_energy = psi_ges(omega, E)

    objective = alignment_energy + lambda_a * global_similarity_energy + lambda_g * local_similarity_energy + lambda_l * ges_energy
    
    return objective
