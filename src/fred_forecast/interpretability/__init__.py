"""Interpretability analysis of NeuroSym-CBF on an already trained checkpoint.

Nothing in this subpackage retrains or modifies the model: everything is a forward pass under
`torch.no_grad()` over frozen weights.
"""
