"""Writes .nlens files, which ``neurolens model.nlens`` opens without any client."""

from .client import _nums

__all__ = ["save_nlens", "save_torch"]


def save_nlens(path, sizes, acts, weights=None, biases=None, labels=None, input=None):
    """weights[l] has shape (sizes[l+1], sizes[l]); biases[l] has sizes[l+1] values."""
    sizes = [int(s) for s in sizes]
    if len(acts) != len(sizes) - 1:
        raise ValueError(f"need {len(sizes) - 1} activation names, got {len(acts)}")
    lines = ["# written by neurolens.save_nlens", "layers " + " ".join(map(str, sizes)), "act " + " ".join(acts)]
    if labels is not None:
        lines.append("labels " + " ".join(str(l).replace(" ", "_") for l in labels))
    for l, w in enumerate(weights or []):
        lines.append(f"w {l} " + _nums(w))
    for l, b in enumerate(biases or []):
        if b is not None:
            lines.append(f"b {l} " + _nums(b))
    if input is not None:
        lines.append("input " + _nums(input))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_torch(model, path, labels=None, input=None):
    """Saves a torch MLP (a stack of nn.Linear) as a .nlens file."""
    from .torch_watch import describe

    layers, sizes, acts = describe(model)
    save_nlens(path, sizes, acts,
             weights=[lin.weight for lin, _ in layers],
             biases=[lin.bias for lin, _ in layers],
             labels=labels, input=input)
