#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.zarrds import ZarrDataset


class AntData(ZarrDataset):
    """
    Frequency Dataset:
    This class is storing Antenna Metadata for UV data sets
    mainly using xarray.Dataset
    """
    # Data type name
    name = "Antenna Dataset"

    # Group Name of zarr file
    zarr_group = "antenna"
