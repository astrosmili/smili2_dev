#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''


class ZarrScanDatasetList(object):
    # Data type name
    name = "zarr_scandataset_list_default"

    # Xarray Dataset
    ds_list = None

    # Group Name of zarr file
    zarr_group_head = "zarr_scandataset_list_default"

    # scan data class
    scandataclass = None

    def __init__(self, ds_list):
        self.ds_list = ds_list

    @classmethod
    def __cls__init__(cls, ds_list):
        return cls(ds_list=ds_list)

    def copy(self):
        """
        Replicate this uvdata set to a new UVData object.

        Returns:
            Replicated data
        """
        import dask
        Nscan = len(self.ds_list)
        ds_list = [
            dask.delayed(self.ds_list[iscan].copy)() for iscan in range(Nscan)
        ]
        return self.__cls__init__(ds_list=dask.compute(*ds_list))

    @classmethod
    def load_zarr(cls, inzarr, Nscan):
        import dask
        ds_list = [dask.delayed(self.scandataclass.load_zarr)(inzarr, iscan)
                   for iscan in range(Nscan)]
        return cls(ds_list=dask.compute(*ds_list))

    def to_zarr(self, outzarr):
        """
        Save to zarr. Dataset of this object will be saved to
        the specified zarr under the group name given by self.zarr_group

        Args:
            outzarr (string): output filename.
        """
        import zarr
        import dask
        z = zarr.open(outzarr, mode="a")
        z.create_group(self.zarr_group_head, overwrite=True)
        Nscan = len(self.ds_list)
        output = [
            dask.delayed(self.ds_list[iscan].to_zarr)(outzarr) for iscan in range(Nscan)
        ]
        dask.compute(*output)
