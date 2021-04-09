#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.zarrds import ZarrDataset


class SrcData(ZarrDataset):
    """
    Source Dataset:
    This class is storing Source Metadata for UV data sets
    mainly using xarray.Dataset
    """
    # Data type name
    name = "Source Dataset"

    # Group Name of zarr file
    zarr_group = "source"