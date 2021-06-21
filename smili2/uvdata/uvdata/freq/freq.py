#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.zarrds import ZarrDataset


class FreqData(ZarrDataset):
    """
    Frequency Dataset:
    This class is storing Frequency Metadata for UV data sets
    mainly using xarray.Dataset
    """
    # Data type name
    name = "Frequency Dataset"

    # Group Name of zarr file
    zarr_group = "frequency"

    def get_shape(self):
        return (self.ds.if_freq.size, self.ds.Nch)

    def recalc_freq(self):
        from numpy import zeros, arange

        # reset index
        ds = self.ds

        # get the number of if and ch
        Nif, Nch = self.get_shape()

        # create an array
        freqarr = zeros([Nif, Nch], dtype="float64")
        chidarr = arange(Nch)

        # compute frequency
        for iif in range(Nif):
            if_freq = ds.if_freq.data[iif]
            sideband = ds.sideband.data[iif]
            ch_bw = ds.ch_bw.data[iif]
            freqarr[iif] = if_freq + sideband * ch_bw * chidarr

        ds["freq"] = (["spw", "ch"], freqarr)
