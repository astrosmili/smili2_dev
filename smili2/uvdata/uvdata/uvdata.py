#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
__author__ = "Smili Developer Team"

# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
# standard modules

# internal
from ...util import warn


class UVData(object):
    # Array infomation
    array = None

    # Source information
    source = None

    # Frequency information
    freq = None

    # complex visibilities
    vis = None

    # pol/frequency independent antenna-based information
    antable = None

    # pol/frequency dependent antenna-based information
    gaintable = None

    # polarization
    poltype = "circ"

    def set_array(self, array):

        from ...array import Array
        if not isinstance(array, Array):
            raise(ValueError())
        self.array = array

    def set_source(self, source):
        '''
        '''
        from ...source import Source
        if not isinstance(source, Source):
            raise(ValueError())
        self.source = source

    def set_freq(self, freq):
        '''
        '''
        from ...freq import Freq
        if not isinstance(freq, Freq):
            raise(ValueError())
        self.freq = freq

    def init_vis(self, utc, dutc=None):
        '''
        Create an empty visibility data set based on the input UTC information.

        Args:
            utc (astropy.time.Time):
                UTC array of data sets
            dutc (astropy.time.TimeDelta, default=None)
                A scalar or array of the integration time
        Returns:
            No returns.
        '''
        from numpy import abs, diff, zeros, isscalar
        from xarray import concat, DataArray
        from itertools import combinations

        # sanity check
        if self.array is None:
            raise ValueError(
                "Please set the array information with set_array() before running this method.")

        if self.freq is None:
            raise ValueError(
                "Please set the frequency setup with set_freq() before running this method.")

        # get antenna ids
        antids = self.array.table.index.tolist()

        # get the number of data
        Nutc = len(utc)
        Nif, Nch = self.freq.get_shape()
        Npol = 4

        # polarizations
        if self.poltype == "circ":
            pols = ["RR", "LL", "RL", "LR"]
        elif self.poltype == "linear":
            pols = ["XX", "YY", "XY", "YX"]
        else:
            pols = ["I", "Q", "U", "V"]

        # get mjd
        mjd = utc.mjd

        # get integration time of each data point
        if dutc is None:
            dutc = abs(diff(utc)).min()
        dmjd = dutc.jd

        if isscalar(dmjd):
            dmjd = [dmjd for i in range(Nutc)]
        elif len(dmjd) != len(mjd):
            raise ValueError(
                "dutc must be scalar or has the same length with utc.")

        # Make Empty Data
        vis_list = []
        for antid1, antid2 in combinations(antids, 2):
            vis_tmp = DataArray(
                zeros([Npol, Nif, Nch, Nutc], dtype="complex128"),
                dims=["pol", "if", "ch", "data"],
                coords=dict(
                    mjd=("data", mjd),
                    dmjd=("data", dmjd),
                    usec=("data", zeros(Nutc, dtype="float64")),
                    vsec=("data", zeros(Nutc, dtype="float64")),
                    wsec=("data", zeros(Nutc, dtype="float64")),
                    antid1=("data", [antid1 for i in range(Nutc)]),
                    antid2=("data", [antid2 for i in range(Nutc)]),
                    flag=(["pol", "if", "ch", "data"], zeros(
                        [Npol, Nif, Nch, Nutc], dtype="int32")),
                    sigma=(["pol", "if", "ch", "data"], zeros(
                        [Npol, Nif, Nch, Nutc], dtype="float64")),
                    pol=(["pol"], pols),
                    freq=(["if", "ch"], self.freq.get_freqarr())
                )
            )
            vis_list.append(vis_tmp)
        self.vis = concat(vis_list, dim="data")

    def init_antable(self, overwrite=False):
        from numpy import unique
        from astropy.time import Time
        from smili2.uvdata.uvdata.antable import ANTable

        mjd_unq, mjdidx_antable = unique(
            self.vis.mjd.values, return_inverse=True)
        utc = Time(mjd_unq, format="mjd", scale="utc")
        self.vis.coords["mjdidx_antab"] = (["data"], mjdidx_antable)

        if self.array is None:
            raise ValueError(
                "Please set an Array object with UVData.set_array()")

        if self.source is None:
            raise ValueError(
                "Please set an Array object with UVData.set_source()")

        if (overwrite is False) and (self.antable is not None):
            warn("Since antable already exists, it won't be recalculated. Use overwrite=True for recalculation.")
            return

        self.antable = ANTable.make(
            utc=utc, array=self.array, source=self.source)

    def copy(self):
        """
        Replicate this uvdata set to a new UVData object.

        Returns:
            UVData: replicated uv data set.
        """
        from copy import deepcopy

        outdata = UVData()

        if self.array is not None:
            outdata.array = self.array.copy()

        if self.source is not None:
            outdata.source = self.source.copy()

        if self.freq is not None:
            outdata.freq = self.freq.copy()

        if self.vis is not None:
            outdata.vis = self.vis.copy()

        if self.antable is not None:
            outdata.antable = self.antable.copy()

        if self.gaintable is not None:
            outdata.gaintable = self.gaintable.copy()

        outdata.poltype = deepcopy(self.poltype)

        return outdata

    def copy_from_antab_to_vis(self,
                               columns=["antname", "x", "y",
                                        "z", "ra", "dec", "az", "el"],
                               overwrite_antable=False):
        from xarray import concat

        # create antable
        self.init_antable(overwrite=overwrite_antable)

        # antenna ids
        antids = self.array.table.index.to_list()

        # categorize the columns
        cols_noant = []
        cols_ant = []
        for col in columns:
            if col in "gst,ra,dec":
                cols_noant.append(col)
            else:
                cols_ant.append(col)

        # Antenna-based information
        vis_list = []
        for antid in antids:
            idx = self.vis.antid1 == antid
            vis_tmp = self.vis.loc[:, :, :, idx]
            if vis_tmp.shape[-1] == 0:
                continue
            antab_tmp = self.antable.query(
                "antid == @antid").reset_index(drop=True)
            for column in cols_noant:
                vis_tmp.coords[column] = (
                    "data", antab_tmp.loc[vis_tmp.coords["mjdidx_antab"].values, column].values)
            for column in cols_ant:
                vis_tmp.coords[column+"1"] = (
                    "data", antab_tmp.loc[vis_tmp.coords["mjdidx_antab"].values, column].values)
            vis_list.append(vis_tmp)
        self.vis = concat(vis_list, dim="data")

        vis_list = []
        for antid in antids:
            idx = self.vis.antid2 == antid
            vis_tmp = self.vis.loc[:, :, :, idx]
            if vis_tmp.shape[-1] == 0:
                continue
            antab_tmp = self.antable.query(
                "antid == @antid").reset_index(drop=True)
            for column in cols_ant:
                vis_tmp.coords[column+"2"] = (
                    "data", antab_tmp.loc[vis_tmp.coords["mjdidx_antab"].values, column].values)
            vis_list.append(vis_tmp)
        self.vis = concat(vis_list, dim="data")

    def calc_uvw_sec(self, overwrite_antable=False):
        from ...util.units import conv, Unit

        # Source and Telescope Locations
        iscolumn = True
        mandatory_columns = "gst,ra,dec,x1,y1,z1,x2,y2,z2".split(",")
        for column in mandatory_columns:
            iscolumn &= column in self.vis.coords
        if not iscolumn:
            self.copy_from_antab_to_vis(columns="gst,ra,dec,x,y,z".split(
                ","), overwrite_antable=overwrite_antable)

        # compute_uvw_sec
        deg2rad = conv(Unit("deg"), Unit("rad"))
        h2rad = conv(Unit("hourangle"), Unit("rad"))

        usec, vsec, wsec = _compute_uvw_sec(
            gst=self.vis.coords["gst"].values*h2rad,
            ra=self.vis.coords["ra"].values*deg2rad,
            dec=self.vis.coords["dec"].values*deg2rad,
            x1=self.vis.coords["x1"].values,
            y1=self.vis.coords["y1"].values,
            z1=self.vis.coords["z1"].values,
            x2=self.vis.coords["x2"].values,
            y2=self.vis.coords["y2"].values,
            z2=self.vis.coords["z2"].values,
        )

        # add results
        self.vis.coords["usec"] = ("data", usec)
        self.vis.coords["vsec"] = ("data", vsec)
        self.vis.coords["wsec"] = ("data", wsec)

    def calc_gst(self):
        '''
        re-calc GST visibilities.
        '''
        from astropy.time import Time
        from numpy import unique

        mjd = self.vis.mjd.values

        # compute gst for the non-redundant set of MJD,
        # and then bring it back to the original DataArray
        mjd_unq, mjd_unq_invidx = unique(mjd, return_inverse=True)
        utc_unq = Time(mjd_unq, format="mjd", scale="utc")
        gst_unq = utc_unq.sidereal_time(
            "apparent", "greenwich", "IAU2006A").hour
        self.vis.coords["gst"] = ("data", gst_unq[mjd_unq_invidx])


def _compute_uvw_sec(gst, ra, dec, x1, y1, z1, x2, y2, z2):
    from astropy.constants import c
    from numpy import cos, sin

    # constants
    c = c.si.value

    # baseline vector
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2

    # cos, sin
    cosdec = cos(dec)
    sindec = sin(dec)
    cosGH = cos(gst - ra)
    sinGH = sin(gst - ra)

    # Earth-rotate baseline vector
    bl_x = cosGH * dx - sinGH * dy
    bl_y = sinGH * dx + cosGH * dy
    bl_z = dz

    # compute projections
    u = bl_y
    v = -bl_x * sindec + bl_z * cosdec
    w = +bl_x * cosdec + bl_z * sindec

    return u/c, v/c, w/c
