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
    # bi-spectram
    bs = None
    # closure amplitudes
    ca = None

    # stokes/frequency independent antenna-based information
    antable = None

    # stokes/frequency dependent antenna-based information
    gaintable = None

    # stokes
    stokestype = "circ"

    # internal flags
    flags = dict(
        recalc_antable=False,
        recalc_uvw_sec=False,
        recalc_uvw=False,
    )

    def set_array(self, array):
        from ...array import Array
        if not isinstance(array, Array):
            raise ValueError()
        self.array = array

    def set_source(self, source):
        '''
        '''
        from ...source import Source
        if not isinstance(source, Source):
            raise ValueError()
        self.source = source

    def set_freq(self, freq):
        '''
        '''
        from ...freq import Freq
        if not isinstance(freq, Freq):
            raise ValueError()
        self.freq = freq

    def set_freq2vis(self):
        if self.freq is None:
            raise ValueError("Frequency data is not set.")

        if self.vis is None:
            raise ValueError("Visibility data is not set")

        self.vis["freq"] = (["if", "ch"], self.freq.get_freqarr())
        self.vis["chbw"] = (["if"], self.freq.table["ch_bw"].values)

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
        from numpy import abs, diff, zeros, isscalar, ones
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
        Nstokes = 4

        # stokesarizations
        if self.stokestype == "circ":
            stokes = ["RR", "LL", "RL", "LR"]
        elif self.stokestype == "linear":
            stokes = ["XX", "YY", "XY", "YX"]
        else:
            stokes = ["I", "Q", "U", "V"]

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
                zeros([Nstokes, Nif, Nch, Nutc], dtype="complex128"),
                dims=["stokes", "if", "ch", "data"],
                coords=dict(
                    mjd=("data", mjd),
                    dmjd=("data", dmjd),
                    usec=("data", zeros(Nutc, dtype="float64")),
                    vsec=("data", zeros(Nutc, dtype="float64")),
                    wsec=("data", zeros(Nutc, dtype="float64")),
                    antid1=("data", [antid1 for i in range(Nutc)]),
                    antid2=("data", [antid2 for i in range(Nutc)]),
                    flag=(["stokes", "if", "ch", "data"], ones(
                        [Nstokes, Nif, Nch, Nutc], dtype="int32")),
                    sigma=(["stokes", "if", "ch", "data"],
                           zeros([Nstokes, Nif, Nch, Nutc], dtype="float64")),
                    stokes=(["stokes"], stokes)
                )
            )
            vis_list.append(vis_tmp)
        self.vis = concat(vis_list, dim="data")

        self.set_freq2vis()

        self.flags["recalc_antable"] = True
        self.flags["recalc_uvw_sec"] = True
        self.flags["recalc_uvw"] = True

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

        if (overwrite is False) and (self.antable is not None) and (self.flags["recalc_antable"] is False):
            warn("Since antable already exists, it won't be recalculated. Use overwrite=True for recalculation.")
            return

        self.antable = ANTable.make(
            utc=utc, array=self.array, source=self.source)
        self.flags["recalc_antable"] = False

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

        outdata.flags = deepcopy(self.flags)
        outdata.stokestype = deepcopy(self.stokestype)

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

    def calc_uvw(self, recalc_uvw_sec=False, overwrite_antable=False):
        # recompute uvw_sec if needed
        if recalc_uvw_sec or self.flags["recalc_uvw_sec"]:
            self.calc_uvw_sec(overwrite_antable=overwrite_antable)

        # scale frequency
        self.vis.coords["u"] = self.vis["freq"] * self.vis["usec"]
        self.vis.coords["v"] = self.vis["freq"] * self.vis["vsec"]
        self.vis.coords["w"] = self.vis["freq"] * self.vis["wsec"]

        # modify flags
        self.flags["recalc_uvw"] = False

    def calc_uvw_sec(self, overwrite_antable=False):
        """
        Re-calculate uvw coordinates

        Args:
            overwrite_antable (bool, defalt=False):
                if True, it will re-calculate frequency independent antenna
                information.
        """
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

        # modify flags
        self.flags["recalc_uvw_sec"] = False
        self.flags["recalc_uvw"] = True

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

    def calc_sigma(self, inplace=True):
        from numpy import sqrt

        # output
        if inplace:
            outuvd = self
        else:
            outuvd = self.copy()

        # Source and Telescope Locations
        iscolumn = True
        mandatory_columns = "sefd1,sefd2".split(",")
        for column in mandatory_columns:
            iscolumn &= column in outuvd.vis.coords
        if not iscolumn:
            outuvd.copy_from_antab_to_vis(
                columns=mandatory_columns, overwrite_antable=True)

        dt = outuvd.vis.dmjd * 86400
        dnu = outuvd.vis.chbw
        factor = 1/sqrt(2*dnu*dt)
        outuvd.vis.sigma[0] = sqrt(outuvd.vis.sefd11 * outuvd.vis.sefd12)
        outuvd.vis.sigma[1] = sqrt(outuvd.vis.sefd21 * outuvd.vis.sefd22)
        outuvd.vis.sigma[2] = sqrt(outuvd.vis.sefd11 * outuvd.vis.sefd22)
        outuvd.vis.sigma[3] = sqrt(outuvd.vis.sefd21 * outuvd.vis.sefd12)
        outuvd.vis["sigma"] = outuvd.vis.sigma * factor

        # return
        if not inplace:
            return outuvd

    def add_thermal_noise(self, inplace=True):
        from numpy.random import normal

        # output
        if inplace:
            outuvd = self
        else:
            outuvd = self.copy()

        # thermal noise
        thnoise = normal(size=outuvd.vis.sigma.shape)+1j * \
            normal(size=outuvd.vis.sigma.shape)
        thnoise = thnoise * outuvd.vis.sigma
        outuvd.vis.data += thnoise.data

        # return
        if not inplace:
            return outuvd

    def apply_flag(self, keep_flagged=False, inplace=False):
        if inplace:
            outdata = self
        else:
            outdata = self.copy()

        idx = outdata.vis.coords["flag"].min(axis=0).min(axis=0).min(axis=0)
        if keep_flagged:
            idx = idx <= 0
        else:
            idx = outdata.vis.coords["flag"].min(
                axis=0).min(axis=0).min(axis=0) >= 0

        outdata.vis = outdata.vis[:, :, :, idx]
        if not inplace:
            return outdata

    def apply_ellimit(self, apply_flag=True, overwrite_antable=False, inplace=True):
        if inplace:
            outdata = self
        else:
            outdata = self.copy()

        # Source and Telescope Locations
        iscolumn = True
        mandatory_columns = "el1,el2".split(",")
        for column in mandatory_columns:
            iscolumn &= column in outdata.vis.coords
        if not iscolumn:
            outdata.copy_from_antab_to_vis(
                columns="el".split(","),
                overwrite_antable=overwrite_antable)

        # antenna ids
        antids = outdata.array.table.index.to_list()

        # Antenna-based information
        for antid in antids:
            elmin, elmax = outdata.array.table.loc[antid, ["elmin", "elmax"]]
            # antid1
            idx = self.vis.coords["el1"] < elmin
            idx |= self.vis.coords["el1"] > elmax
            idx &= self.vis.coords["antid1"] == antid
            outdata.vis.flag[:, :, :, idx] = -1
            # antid2
            idx = self.vis.coords["el2"] < elmin
            idx |= self.vis.coords["el2"] > elmax
            idx &= self.vis.coords["antid2"] == antid
            outdata.vis.flag[:, :, :, idx] = -1

        if apply_flag:
            outdata.apply_flag(inplace=True)

        if not inplace:
            return outdata

    def eval(self, input, inplace=False):
        """
        [summary]

        Args:
            input ([type]): [description]
            inplace (bool, optional): [description]. Defaults to False.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        from ...imdata import Image

        out = None
        if isinstance(input, Image):
            out = self.eval_image(image=input, inplace=inplace)
        else:
            raise ValueError("Invalid input data type")

        if not inplace:
            return out

    def eval_image(self, image, inplace=False):
        """
        [summary]

        Args:
            image ([type]): [description]
            inplace (bool, optional): [description]. Defaults to False.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        from ...ft.ft_image import NFFT_Image
        from numpy import zeros

        # get the image dimension
        ni_t, ni_f, ni_s, ny, nx = image.data.shape
        nim = ni_t*ni_f

        # get the data dimension
        nd_s, nif, nch, ndata = self.vis.shape

        # the number of images
        if nim != 1:
            raise ValueError(
                "Currently this function works for the two dimensional image")

        # get uv coordinates
        u = self.vis.u.data.flatten()
        v = self.vis.v.data.flatten()
        ntotal = len(u)

        # initialize NFFT function
        nfft = NFFT_Image(
            u=u, v=v,
            dx=-image.meta["dx"].val,
            dy=image.meta["dy"].val,
            nx=nx,
            ny=ny
        )

        # generate model visibilities in IQUV
        vis_mod = zeros([nd_s, ntotal], dtype="complex128")
        for ipol in range(ni_s):
            vis_mod[ipol] = nfft.nfft2d_forward(image.data[0, 0, ipol].data)

        # apply pulse function
        pulsekernel = image.get_pulsefunc()(u, v)
        vis_mod[:] *= pulsekernel

        # output
        if inplace:
            outuvd = self
        else:
            outuvd = self.copy()

        # change polarization
        if outuvd.stokestype == "stokes":
            outuvd.vis.data = vis_mod.reshape([nd_s, nif, nch, ndata])
        elif outuvd.stokestype == "circ":
            rrvis = vis_mod[0] + vis_mod[3]  # RR = I+V
            llvis = vis_mod[0] - vis_mod[3]  # LL = I-V
            rlvis = vis_mod[1] + 1j * vis_mod[2]  # RL = Q+iU
            lrvis = vis_mod[1] - 1j * vis_mod[2]  # LR = Q-iU
            outuvd.vis.data[0] = rrvis.reshape([nif, nch, ndata])
            outuvd.vis.data[1] = llvis.reshape([nif, nch, ndata])
            outuvd.vis.data[2] = rlvis.reshape([nif, nch, ndata])
            outuvd.vis.data[3] = lrvis.reshape([nif, nch, ndata])
        elif outuvd.stokestype == "linear":
            xxvis = vis_mod[0] + vis_mod[1]  # XX = I+Q
            yyvis = vis_mod[0] - vis_mod[1]  # YY = I-Q
            xyvis = vis_mod[2] + 1j * vis_mod[3]  # XY = U+iV
            yxvis = vis_mod[2] - 1j * vis_mod[3]  # YX = U-iV
            outuvd.vis.data[0] = xxvis.reshape([nif, nch, ndata])
            outuvd.vis.data[1] = yyvis.reshape([nif, nch, ndata])
            outuvd.vis.data[2] = xyvis.reshape([nif, nch, ndata])
            outuvd.vis.data[3] = yxvis.reshape([nif, nch, ndata])

        # return
        if not inplace:
            return outuvd

    @classmethod
    def load_uvfits(cls, uvfits, printlevel=0):
        """
        Load an uvfits file. Currently, this function can read only single-source,
        single-frequency-setup, single-array data correctly.

        Args:
            uvfits (string or pyfits.HDUList object): input uvfits data
            printlevel (integer): print some notes. 0: silient 3: maximum level
        Returns:
            uvdata.UVData object
        """
        from .io.uvfits import uvfits2UVData
        return uvfits2UVData(uvfits=uvfits, printlevel=printlevel)


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
