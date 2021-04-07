#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from .zarrds import ZarrDataset


class ZarrScanDataset(ZarrDataset):
    # Data type name
    name = "zarr_scandataset_default"

    # Data Set
    #   This supposed to include the data set
    ds = None

    # Group Name of zarr file
    zarr_group_head = "zarr_scandataset_default"

    def __init__(self, ds):
        """
        Initialize the instance.

        Args:
            ds (xarray.Dataset): Input Dataset
        """
        self.ds = ds.copy()
        self.zarr_group = self.zarr_group_head+"/%04d" % (ds.scanid)

    @classmethod
    def load_zarr(cls, inzarr, scanid):
        """
        Open a dataset from the specified zarr file.
        The dataset will be loaded from the group given by cls.zarr_group.

        Args:
            inzarr (string): input zarr file.

        Returns:
            Loaded object
        """
        from xarray import open_zarr
        ds = open_zarr(inzarr, group=cls.zarr_group_head + "/%04d" % (scanid))
        return cls(ds=ds)

    def to_zarr(self, outzarr):
        """
        Save to zarr. Dataset of this object will be saved to
        the specified zarr under the group name given by self.zarr_group

        Args:
            outzarr (string): output filename.
        """
        self.ds.to_zarr(outzarr, mode="a", group=self.zarr_group)
