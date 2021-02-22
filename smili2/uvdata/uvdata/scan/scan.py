#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
from ....util.zarrds import ZarrDataset


class ScanData(ZarrDataset):
    """
    Scan Dataset:
    This class is storing Scan Metadata for UV data sets
    mainly using xarray.Dataset
    """
    # Data type name
    name = "Scan Dataset"

    # Group Name of zarr file
    zarr_group = "scan"
