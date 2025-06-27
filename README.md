This repository hosts the source code and data‐generation utilities for UHFreq.  
The project is organised as follows:
| Path | Purpose |
|------|---------|
| **`complexModules.py`** | Implements high‑level building blocks.  In particular, `FrequencyRepresentationModule_skiplayer32` **is the full UHFreq framework** responsible for complex‑domain spectral representation. |
| **`complexLayers.py`** | Collection of auxiliary **complex‑valued neural‑network layers** (e.g., complex convolution, batch normalisation, and activation functions) that are reused across modules. |
| **`complexFunctions.py`** | **Mathematical utilities** for complex arithmetic, phase handling, and polar/Cartesian conversions.  These functions support `complexLayers.py` and UHFreq internals. |
