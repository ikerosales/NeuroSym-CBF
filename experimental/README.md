# Experimental

Código de trabajo preliminar/futuro — **no** evaluado formalmente como parte de los
resultados reportados en la memoria del TFG, y **no** forma parte del pipeline reproducible
principal (`src/fred_forecast/` + `scripts/`).

## Qué hay aquí

- **`neurosym_cbf_pinball/`** — variante de NeuroSym-CBF con pérdida pinball para forecasting
  probabilístico (cuantiles 0.5 y 0.9). La arquitectura y el bucle de entrenamiento están
  migrados y verificados con datos sintéticos (mismo nivel de fidelidad que el resto del
  repo), pero no se llegó a evaluar en profundidad — se dejó como línea de trabajo futuro.
  Reutiliza los componentes compartidos y ya validados de `src/fred_forecast/` (dataset,
  patch encoder, TopKSAE, métricas base) — aquí solo vive lo específico de la variante
  probabilística (cabeza simbólica y FiLM vectorizados por cuantil, métricas de pinball/
  cobertura/anchura de intervalo).
- **`chronos2_pinball/`** — contraparte zero-shot de lo anterior: evalúa Chronos-2 en modo
  probabilístico (cuantiles 0.5 y 0.9 vía `torch.quantile` sobre las muestras que devuelve el
  pipeline), con las mismas métricas de pinball/cobertura/anchura que `neurosym_cbf_pinball`
  (reutilizadas de ahí, no reimplementadas). Pensado específicamente para comparar contra
  NeuroSym-CBF (pinball) — ver `experimental/evaluate_chronos2_pinball.py`.

## Por qué está separado

Para que quien navegue el repo distinga a simple vista qué es el trabajo validado del TFG
(`src/fred_forecast/`) de qué es una línea abierta/sin terminar. El código de aquí no se
instala como parte del paquete `fred-forecast` (no está bajo `src/`) y no tiene un script
numerado en `scripts/` — se ejecuta aparte, ver `experimental/train_neurosym_cbf_pinball.py`.
