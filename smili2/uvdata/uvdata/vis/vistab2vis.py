#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
import dask


def vistab2visds(uvd, remove_vistab=False):
    """
    Split vistable into scan-based visibility data sets.

    Args:
        uvd (uvdata.UVData): 
            Input UVData Object
        remove_vistab (bool, optional): 
            If True, vistab will be removed after generating vis data set.
            Defaults to False.
    """
    from .vis import VisData
    vistabds = uvd.vistab.ds
    output_list = []

    # Form visibility data set for each scan
    for scangroup in vistabds.groupby("scanid"):
        output_list.append(
            vistabscan2visdsscan(scangroup, outzarr=uvd.zarrfile)
        )
    uvd.vis = VisData(ds_list=dask.compute(*output_list))

    # remove vistab
    if remove_vistab:
        uvd.vistab = None


@dask.delayed
def vistabscan2visdsscan(scangroup, outzarr):
    from xarray import Dataset
    from numpy import complex128, float64, int32, unique, zeros
    from dask.array import arange
    from .visscan import VisScanData

    # get id and dataset
    scanid = scangroup[0]
    scands = scangroup[1]

    # get data number
    Ndata, Nif, Nch, Nstokes = scands.vis.shape
    del Ndata

    # MJD index
    mjdset, mjdidx, mjdinv = unique(
        scands.mjd, return_index=True, return_inverse=True)
    Nt = mjdset.size
    scands.coords["tidx"] = ("data", arange(Nt)[mjdinv])

    # baseline index
    blid = scands.antid1*10000 + scands.antid2
    blset, blidx, blinv = unique(blid, return_index=True, return_inverse=True)
    Nb = blset.size
    scands.coords["bidx"] = ("data", arange(Nb)[blinv])

    # create data set
    visds_scan = VisScanData(
        Dataset(
            data_vars=dict(
                vis=(["time", "baseline", "if", "ch", "stokes"], zeros(
                    [Nt, Nb, Nif, Nch, Nstokes], dtype=complex128))
            ),
            coords=dict(
                mjd=("time", mjdset),
                dmjd=("time", scands.dmjd.data[mjdidx]),
                antid1=("baseline", scands.antid1.data[blidx]),
                antid2=("baseline", scands.antid2.data[blidx]),
                usec=(["time", "baseline"], zeros((Nt, Nb), dtype=float64)),
                vsec=(["time", "baseline"], zeros((Nt, Nb), dtype=float64)),
                wsec=(["time", "baseline"], zeros((Nt, Nb), dtype=float64)),
                sigma=(["time", "baseline", "if", "ch", "stokes"],
                       zeros((Nt, Nb, Nif, Nch, Nstokes), dtype=float64)),
                flag=(["time", "baseline", "if", "ch", "stokes"],
                      zeros((Nt, Nb, Nif, Nch, Nstokes), dtype=int32))
            ),
            attrs=dict(
                scanid=scanid
            )
        )
    )

    for mjdgroup in scands.groupby("tidx"):
        tidx = mjdgroup[0]
        for blgroup in mjdgroup[1].groupby("bidx"):
            bidx = blgroup[0]
            bds = blgroup[1]
            visds_scan.ds.vis.data[tidx, bidx] = bds.vis.data
            visds_scan.ds.sigma.data[tidx, bidx] = bds.sigma.data
            visds_scan.ds.flag.data[tidx, bidx] = bds.flag.data
            visds_scan.ds.usec.data[tidx, bidx] = bds.usec.data
            visds_scan.ds.vsec.data[tidx, bidx] = bds.vsec.data
            visds_scan.ds.wsec.data[tidx, bidx] = bds.wsec.data

    visds_scan.to_zarr(outzarr)
    visds_scan = VisScanData.load_zarr(
        inzarr=outzarr, scanid=visds_scan.ds.scanid)
    return visds_scan
