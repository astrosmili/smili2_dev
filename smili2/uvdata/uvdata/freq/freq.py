#!/usr/bin/env python
# -*- coding: utf-8 -*-


class FreqData(object):
    # Xarray Dataset
    ds = None

    def __init__(self, ds):
        self.ds = ds.copy()

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

        ds["freq"] = (["if", "ch"], freqarr)

    def copy(self):
        """
        Replicate this uvdata set to a new UVData object.

        Returns:
            Replicated data
        """

        outdata = FreqData(ds=self.ds.copy())

        return outdata
