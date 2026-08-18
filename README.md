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

Resolvit generates diagnostic plots and residual tables for each channel-filter-window combination. For example, a run with the default parameters produces the following files:

```text
resolvit_data_products/
└── diagnostics/
    └── <product_id>/
        ├── residuals_iteration_1.txt
        ├── residuals_iteration_1.png
        ├── residuals_iteration_2.txt
        ├── residuals_iteration_2.png
        ├── residuals_iteration_3.txt
        ├── residuals_iteration_3.png
        ├── residuals_iteration_4.txt
        ├── residuals_iteration_4.png
        ├── <iteration>_<bin_number>_<bin_mid_time>_correlations.png
        ├── ...
        └── resolvit.log
```

The correlation files follow the naming convention:

```text
<iteration>_<bin_number>_<bin_mid_time>_correlations.png
```

For example:

```text
1_100_418441843_correlations.png
```

corresponds to **iteration 1**, **temporal bin 100**, with a bin midpoint time of `418441843` seconds in the UVIT event-list time system.

The correlation plots show the X and Y correlation functions used to determine the residual shift for the corresponding temporal bin.

### `residuals_iteration_N.txt`

The `residuals_iteration_N.txt` file contains the residual drift estimates obtained during iteration `N`. Each row corresponds to a temporal bin used for residual drift estimation.

The file contains six columns:

```text
t_start t_end total_events events_after_bkg_removal dx dy
```

where:

* `t_start` — start time of the temporal bin, in the UVIT event-list time system. To convert this value to MJD:

  ```text
  MJD = (t_start / 86400) + 55197
  ```

* `t_end` — end time of the temporal bin, in the same time system.

* `total_events` — total number of events in the temporal bin, including both source and background events.

* `events_after_bkg_removal` — total number of events in the temporal bin, after removal of background events.

* `dx` — measured residual shift in the detector X direction, in sub-pixels.

* `dy` — measured residual shift in the detector Y direction, in sub-pixels.

For example:

```text
# t_start t_end total_events events_after_bkg_removal dx dy
418441693.275282 418441793.275282 80530 68421 0.000000 0.000000
418441793.275282 418441893.275282 81233 69104 0.404955 0.052406
418441893.275282 418441993.275282 85853 73128 0.633635 0.233444
418463893.275282 418463993.275282 78350 66742 1.291091 1.110052
...
```

Only temporal bins containing more than

```text
total_events_fraction × median(total_events)
```

events are used for residual estimation. The default value of `total_events_fraction` is `0.75`.

The first accepted temporal bin is used as the reference bin and therefore has:

```text
dx = 0
dy = 0
```

For subsequent bins, `dx` and `dy` represent the measured shift relative to this reference, as determined by image cross-correlation.

The corresponding `residuals_iteration_N.png` file shows `dx` and `dy` as a function of the midpoint time of each temporal bin.

### Iterations

Resolvit performs multiple residual-drift estimation iterations using different temporal offsets. This reduces the sensitivity of the residual estimates to the boundaries of the temporal bins.

With the default parameters, four iterations are performed with offsets:

```text
[0, 1/2, 1/4, 1/3]
```

of the specified `bin_size`.

The diagnostic files therefore correspond to:

* `residuals_iteration_1.*` — residuals obtained from the initial temporal binning with zero offset.
* `residuals_iteration_2.*` — residuals obtained using an offset of `1/2 × bin_size`.
* `residuals_iteration_3.*` — residuals obtained using an offset of `1/4 × bin_size`.
* `residuals_iteration_4.*` — residuals obtained using an offset of `1/3 × bin_size`.

The residual correction from each iteration is applied to the events before the next iteration.

The `<iteration>_<bin_number>_<bin_mid_time>_correlations.png` files show the detailed correlation functions used to measure the X and Y shifts for individual temporal bins. In contrast, the `residuals_iteration_N.png` files provide an overview of the residual drift measured throughout the observation for a given iteration.
