"""Análisis interpretativo de NeuroSym-CBF sobre un checkpoint ya entrenado.

Nada de este subpaquete re-entrena ni modifica el modelo: todo son pases hacia delante con
`torch.no_grad()` sobre pesos congelados.
"""
