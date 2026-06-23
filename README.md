# Resolvit

Resolvit improves the PSF of UVIT Level2 products.

## Installation

```bash
pip install resolvit
```

## Command line usage

```bash
resolvit 20160101_A01_123T01_0123456789_level2
```

## Python usage

```python
from resolvit import process_observation

process_observation(
    "20160101_A01_123T01_0123456789_level2"
)
```
