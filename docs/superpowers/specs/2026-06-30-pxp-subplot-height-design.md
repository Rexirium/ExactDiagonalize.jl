# PXP Spectrum Subplot Height Design

## Goal

Increase the height of the spectrum subplot while keeping the two time-series
subplots equal in height.

## Design

Use a height ratio of `1.3:1:1` for the three visible subplots. Preserve the
existing nested-grid structure by changing the outer grid ratio from `(1, 2)`
to `(1.3, 2)`. The lower grid remains split evenly into two rows, so the second
and third subplots retain equal heights.

The figure size, constrained layout, shared x-axis behavior, plotted data, and
axis formatting remain unchanged.

## Verification

Confirm that the outer grid uses `height_ratios=(1.3, 2)`, the lower subgrid
still uses two equal rows, and the Python file passes syntax compilation.
