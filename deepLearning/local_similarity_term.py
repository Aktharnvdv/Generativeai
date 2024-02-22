def local_similarity_term(vertices, edges, similarity_transformations):
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
