# Resolvit

Resolvit improves the point spread function (PSF) of UVIT Level2 products by applying sub-pixel corrections to the Level2 events list and generating a new set of derived data products.

The corrected events list is the primary Resolvit product. All other Resolvit products are derived from it.

**Note:** Resolvit supports only UVIT Level2 products generated with UVIT Pipeline Version 7 or later. Earlier pipeline products are not supported.

## Requirements

- Python 3.9 or later
- UVIT Level2 products generated with UVIT Pipeline Version 7 or later

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
)
```

`bin_size` specifies the temporal bin size, in seconds, used to estimate residual image shifts.

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

Show the installed version:

```bash
resolvit --version
```

## Method

Resolvit divides the events list into temporal bins and measures residual image shifts between bins using image cross-correlation. The measured shifts are applied as sub-pixel corrections to individual photon events. Multiple iterations with different temporal offsets are performed to reduce bin-edge effects.

## Output products

Resolvit creates a new directory alongside the original UVIT products:

```text
uvit/
├── data_products/
└── resolvit_data_products/
```

For each UVIT channel-filter-window combination, Resolvit generates:

```text
AS1..._l2ce.fits      # Corrected events list

AS1...I_l2img.fits    # Instrument-coordinate count-rate image
AS1...I_l2err.fits    # Instrument-coordinate count-rate error image
AS1...I_l2exp.fits    # Instrument-coordinate exposure map

AS1...A_l2img.fits    # Astronomical-coordinate count-rate image
AS1...A_l2err.fits    # Astronomical-coordinate count-rate error image
AS1...A_l2exp.fits    # Astronomical-coordinate exposure map
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

