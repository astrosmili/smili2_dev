#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''


class ZarrDataset(object):
    # Data type name
    name = "zarr_dataset_default"

    # Data Set
    #   This supposed to include the data set
    ds = None

    # Group Name of zarr file
    zarr_group = "zarr_dataset_default"

    def __init__(self, ds):
        """
        Initialize the instance.

        Args:
            ds (xarray.Dataset): Input Dataset
        """
        self.ds = ds.copy()

    def _repr_html_(self):
        return self.ds._repr_html_()

    @classmethod
    def __cls__init__(cls, ds):
        return cls(ds=ds)

    @classmethod
    def load_zarr(cls, inzarr):
        """
        Open a dataset from the specified zarr file.
        The dataset will be loaded from the group given by cls.zarr_group.

        Args:
            inzarr (string): input zarr file.

        Returns:
            Loaded object
        """
        from xarray import open_zarr

        # print(cls.zarr_group)
        
        ds = open_zarr(inzarr, group=cls.zarr_group)
        return cls(ds=ds)

    def copy(self):
        """
        Replicate this object.

        Returns:
            Replicated data
        """
        return self.__cls__init__(ds=self.ds.copy())

    def chunk(self, **args):
        self.ds = self.ds.chunk(**args)

    def to_zarr(self, outzarr, reload=False):
        """
        Save to zarr. Dataset of this object will be saved to
        the specified zarr under the group name given by self.zarr_group

        Args:
            outzarr (string): output filename.
            reload (bool):
                If true, the dataset will be reloaded from saved file.
                This will allow to convert existing arrays to dask arrays.
        """
        from xarray import open_zarr

        self.ds.to_zarr(outzarr, mode="a", group=self.zarr_group)

        if reload:
            self.ds = open_zarr(outzarr, group=self.zarr_group)
