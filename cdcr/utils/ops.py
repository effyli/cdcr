import torch
import torch.nn.functional as F


def sum_to_int(tensor):
    return int(tensor.float().sum().item())


def safe_div(a, b):
    return a / b if b else 0


def stack_with_padding(tensors, dim=0, pad_value=0):
    """
    Stacks a list of tensors by padding each dimension (on the right side) to the maximum between all tensors
    Supports sparse tensors (albeit with only pad_value=0)

    Args:
        tensors: the lists of (sparse) tensors with D dimensions to stack
        dim: the axis to stack the tensors
        pad_value: the value to use as padding
    Returns:
        a tensor with D+1 dimensions
    """
    assert len(set([t.dim() for t in tensors])) == 1, "tensors must have the same number of dims"

    # get maximum size for each dimension
    max_dims = [0 for _ in range(tensors[0].dim())]
    for tensor in tensors:
        for i, d in enumerate(tensor.size()):
            max_dims[i] = max(max_dims[i], d)

    padded_tensors = []
    for tensor in tensors:
        # if tensor is sparse, padding is just changing the dense shape
        if isinstance(tensor, (torch.sparse.FloatTensor, torch.sparse.LongTensor)):
            padded_tensor = torch.sparse.FloatTensor(
                tensor._indices(),
                tensor._values(),
                max_dims)
        # else we can use the `F.pad` function
        else:
            padding = []
            # since `F.pad` takes padding from inner dimensions to outer dimensions
            # we do it in the opposite direction and reverse
            for i, d in enumerate(max_dims):
                padding.append(d - tensor.size(i))
                padding.append(0)
            padding.reverse()
            padded_tensor = F.pad(tensor, padding, "constant", pad_value)
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors, dim=dim)
