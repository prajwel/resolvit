# Resolvit

Resolvit improves the point spread function (PSF) of UVIT Level2 products by applying sub-pixel corrections to the Level2 events list and generating a new set of derived data products.

The corrected events list is treated as the primary Resolvit product. All other Resolvit products are derived from it.

## Installation

```bash
pip install resolvit
```

## Python usage

```python
from resolvit import process_observation

process_observation(
    "20160101_A01_123T01_0123456789_level2"
)
```

### Custom parameters

```python
from resolvit import process_observation

process_observation(
    "20160101_A01_123T01_0123456789_level2",
    bin_size=50,
    total_events_fraction=0.6,
)
```

## Command-line usage

Process a UVIT Level2 observation directory:

```bash
resolvit 20160101_A01_123T01_0123456789_level2
```

Specify a custom time bin size:

```bash
resolvit 20160101_A01_123T01_0123456789_level2 \
    --bin-size 50
```

Specify a custom event fraction threshold:

```bash
resolvit 20160101_A01_123T01_0123456789_level2 \
    --event-fraction 0.6
```

Show the installed version:

```bash
resolvit --version
```

## Output products

Resolvit creates a new directory alongside the original UVIT products:

```text
uvit/
├── data_products/
└── resolvit_data_products/
```

The original UVIT products are never modified.

## Diagnostics

Resolvit generates diagnostic plots and residual tables for each channel-filter-window combination:

```text
resolvit_data_products/
└── diagnostics/
    └── <product_id>/
        ├── resolvit.log
        ├── residuals_iteration_1.txt
        ├── residuals_iteration_1.png
        ├── residuals_iteration_2.txt
        ├── residuals_iteration_2.png
        └── *_correlations.png
```
