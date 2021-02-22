#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ....util.zarrds import ZarrScanDataset


class VisScanData(ZarrScanDataset):
    # Xarray Dataset
    ds = None

    # Group Name of zarr file
    zarr_group_head = "visibility"
