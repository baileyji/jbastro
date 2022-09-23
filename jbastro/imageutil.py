#!/usr/bin/env python
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans


def median_patch(x, mask, size=6, copy=False):
    if copy:
        x = x.copy()
    x[mask] = np.nan

    # create a "fixed" image with NaNs replaced by interpolated values
    a=interpolate_replace_nans(x, Gaussian2DKernel(x_stddev=size/10))
    a[np.isnan(a)]=0
    return a
