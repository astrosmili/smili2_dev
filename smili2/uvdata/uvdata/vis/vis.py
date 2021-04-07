#!/usr/bin/env python
# -*- coding: utf-8 -*-


from ....util.zarrds import ZarrScanDatasetList


class VisData(ZarrScanDatasetList):
    # Name
    name = "visibility"

    # Xarray Dataset
    ds_list = None

    # Group Name of zarr file
    zarr_group_head = "visibility"

    # Class
    from .visscan import VisScanData as scandataclass
