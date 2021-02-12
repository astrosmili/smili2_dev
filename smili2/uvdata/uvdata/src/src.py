#!/usr/bin/env python
# -*- coding: utf-8 -*-


class SrcData(object):
    # Xarray Dataset
    ds = None

    def __init__(self, ds):
        self.ds = ds.copy()

    def copy(self):
        """
        Replicate this uvdata set to a new UVData object.

        Returns:
            Replicated data
        """

        outdata = SrcData(ds=self.ds.copy())

        return outdata
