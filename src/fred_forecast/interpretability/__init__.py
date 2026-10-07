"""Interpretability analyses of NeuroSym-CBF on an already trained checkpoint.

Nothing in this subpackage retrains or modifies the model: everything is forward passes with
`torch.no_grad()` over frozen weights.
"""
