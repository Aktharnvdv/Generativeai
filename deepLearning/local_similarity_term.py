def local_similarity_term(vertices, edges, similarity_transformations):
    """
    Calculate the energy of the local similarity term based on the given vertices, edges, and similarity transformations.

    Args:
    - vertices (torch.Tensor): Tensor representing vertex positions (N x D).
    - edges (list): List of edges [(j, k)].
    - similarity_transformations (dict): Dictionary {edge: Sjk} where Sjk is a D x D similarity transformation matrix.

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