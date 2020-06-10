#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a sub-module of smili2 handling image data.
'''
__author__ = "Smili Developer Team"


class Image(object):
    '''
    The Class to handle five dimensional images

    Attributes:
        data (xarray.DataArray): Image data
        meta (util.metadata.MetaData): Meta data
        angunit (str): Angular Unit
    '''
    # attributes
    data = None
    meta = None

    # angunit
    angunit = "uas"

    def __init__(self,
                 dx=2, dy=None,
                 nx=128, ny=None,
                 nxref=None, nyref=None,
                 angunit="uas",
                 mjd=[0.],
                 freq=[230e9],
                 source=None,
                 srccoord=None,
                 instrument=None,
                 **args):
        """
        The Class to handle five dimensional images

        Args:
            dx (int, optional): [description]. Defaults to 2.
            dy ([type], optional): [description]. Defaults to None.
            nx (int, optional): [description]. Defaults to 128.
            ny ([type], optional): [description]. Defaults to None.
            nxref ([type], optional): [description]. Defaults to None.
            nyref ([type], optional): [description]. Defaults to None.
            angunit (str, optional): [description]. Defaults to "uas".
            mjd (list, optional): [description]. Defaults to [0.].
            freq (list, optional): [description]. Defaults to [230e9].
            source (str): The source name.
            srccoord (astropy.coordinates.SkyCoord):
                Coordinates of the source. If not specified,
            instrument (str, optional): The name of instrument, telescope.

        Attributes:
            data (xarray.DataArray): Image data
            meta (util.metadata.MetaData): Meta data
            angunit (str): Angular Unit
        """
        from numpy import float64, int32, abs
        from ..util.units import conv

        # Parse arguments
        #  dx & dy
        factor = conv(angunit, "rad")
        args["dx"] = abs(float64(dx*factor))
        if dy is None:
            args["dy"] = args["dx"]
        else:
            args["dy"] = abs(float64(dy*factor))

        #  nx & ny
        args["nx"] = int32(nx)
        if ny is None:
            args["ny"] = args["nx"]
        else:
            args["ny"] = int32(ny)

        # nxref & nyref
        if nxref is None:
            args["nxref"] = args["nx"]/2 - 0.5
        else:
            args["nxref"] = float64(nxref)
        if nyref is None:
            args["nyref"] = args["ny"]/2 - 0.5
        else:
            args["nyref"] = float64(nyref)

        #  angunit
        self.angunit = angunit

        # initialize meta data
        self._init_meta()
        for key in args.keys():
            if key not in self.meta.keys():
                raise ValueError("Key '{}' is not in the meta.".format(key))
            elif key in ["x", "y"]:
                raise ValueError(
                    "Use srccoord for the reference pixel coordinates x and y.")

            self.meta[key].val = args[key]

        # initialize data
        self.set_source(source, srccoord)
        self.set_instrument(instrument)
        self.init_data()
        self.init_stokes()
        self.init_xygrid()
        self.set_mjd(mjd)
        self.set_freq(freq)

    def _init_meta(self):
        '''
        Initialize the metadata.
        '''
        from ..util.metadata import MetaData, MetaRec
        self.meta = MetaData(
            # Source
            source=MetaRec(val="No Name", unit="",
                           dtype="str", comment="Source Name"),
            instrument=MetaRec(val="No Name", unit="",
                               dtype="str", comment="Instrument's Name"),
            # RA
            x=MetaRec(val=0., unit="rad", dtype="float64",
                      comment="RA of the reference pixel"),
            dx=MetaRec(val=1., unit="rad", dtype="float64",
                       comment="Pixel size in RA axis"),
            nxref=MetaRec(val=0., unit="", dtype="float64",
                          comment="Pixel ID of the reference pixel in RA axis"),
            nx=MetaRec(val=1, unit="", dtype="int32",
                       comment="Number of pixels in RA axis"),
            # Dec
            y=MetaRec(val=0., unit="rad", dtype="float64",
                      comment="Dec of the reference pixel"),
            dy=MetaRec(val=1., unit="rad", dtype="float64",
                       comment="Pixel size in Dec direction"),
            nyref=MetaRec(val=0., unit="", dtype="float64",
                          comment="Pixel ID of the reference pixel in Dec axis"),
            ny=MetaRec(val=64, unit="", dtype="int32",
                       comment="Number of pixels in Dec axis"),
            # Stokes/Polarization
            ns=MetaRec(val=1, unit="", dtype="int32",
                       comment="Number of pixels in Stokes axis"),
            # Frequency
            nf=MetaRec(val=1, unit="", dtype="int32",
                       comment="Number of pixels in Freq. axis"),
            # Time
            nt=MetaRec(val=1, unit="", dtype="int32",
                       comment="Number of pixels in Time axis"),
            # Pulse Function
            ptype=MetaRec(val="rect", unit="", dtype="str",
                          comment="Pixel Pulse Functions"),
            # Beam size
            bmaj=MetaRec(val=0., unit="rad", dtype="float64",
                         comment="Major-axis Beam Size"),
            bmin=MetaRec(val=0., unit="rad", dtype="float64",
                         comment="Minor-axis Beam Size"),
            bpa=MetaRec(val=0., unit="rad", dtype="float64",
                        comment="Beam Position Angle"),
        )

    def init_data(self):
        '''
        Initialize the image data array. This creates an empty image with the
        size specified in the metadata.
        '''
        from xarray import DataArray
        from numpy import zeros, float64

        # read the array size
        nt = self.meta["nt"].val
        nf = self.meta["nf"].val
        ns = self.meta["ns"].val
        ny = self.meta["ny"].val
        nx = self.meta["nx"].val

        # array shape
        shape = (nt, nf, ns, ny, nx)

        # create data
        self.data = DataArray(
            data=zeros(shape, dtype=float64),
            dims=["mjd", "freq", "stokes", "y", "x"]
        )

    def init_stokes(self):
        '''
        Add the stokes information to the image data array.
        '''
        nf = self.meta["nf"].val

        if nf == 1:
            self.data["stokes"] = ["I"]
        elif nf == 4:
            self.data["stokes"] = ["I", "Q", "U", "V"]
        else:
            raise ValueError(
                "Current version of SMILI accepts only single or full polarization images.")

    def init_xygrid(self):
        '''
        Add the xy coordinates (in radians) to the image data array.
        '''
        from numpy import arange

        sign_dx = dict(x=-1, y=1)
        for axis in ["x", "y"]:
            # get the meta info
            nx = self.meta["n"+axis].val
            dx = self.meta["d"+axis].val
            nxref = self.meta["n{}ref".format(axis)].val
            sign = sign_dx[axis]

            # compute coordinates
            self.data[axis] = sign*dx*(arange(nx)-nxref)

    def set_source(self, source="M87", srccoord=None):
        '''
        Set the source name and the source coordinate to the metadata.
        If source coordinate is not given, it will be taken from the CDS.

        Args:
            source (str; default="SgrA*"):
                Source Name
            srccoord (astropy.coordinates.Skycoord object; default=None):
                Source position. If not specified, it is automatically pulled
                from the CDS.
        '''
        from astropy.coordinates import SkyCoord

        if source is not None:
            self.meta["source"].val = source
            if srccoord is None:
                srccoord = SkyCoord.from_name(source)

        if srccoord is not None:
            self.meta["x"].val = srccoord.ra.rad
            self.meta["y"].val = srccoord.dec.rad

    def set_instrument(self, instrument="EHT"):
        """
        Set the metadata for the instrument with a
        specified name of the instrument.

        Args:
            instrument (str): The instrument name. Defaults to "EHT".
        """
        if instrument is not None:
            self.meta["instrument"].val = instrument

    def set_mjd(self, mjd):
        """
        Set the MJD infromation for the image data array.

        Args:
            mjd (float or array): MJD
        """
        from numpy import isscalar, asarray

        if isscalar(mjd):
            mjd_arr = asarray([mjd], dtype="float64")
        else:
            mjd_arr = mjd.copy()

        self.data["mjd"] = mjd_arr

    def set_freq(self, freq):
        """
        Set the frequency infromation for the image data array.

        Args:
            freq (float or array): Frequency in Hz
        """
        from numpy import isscalar, asarray

        if isscalar(freq):
            freq_arr = asarray([freq], dtype="float64")
        else:
            freq_arr = freq.copy()

        self.data["freq"] = freq_arr

    def set_beam(self, majsize=0., minsize=0., pa=0., scale=1., angunit=None):
        '''
        Set beam parameters into headers.

        Args:
            majsize, minsize(float, default=0):
                major/minor-axis FWHM size
            scale(float, default=1):
                scaling factor that will be multiplied to maj/min size.
            pa(float, default=0):
                position angle in deg
        '''
        from ..util.units import conv
        from numpy import deg2rad

        # get the conversion factor for the angular unit
        if angunit is None:
            angunit = self.angunit
        factor = conv(angunit, "rad")

        self.meta["bmaj"].val = majsize * factor * scale
        self.meta["bmin"].val = minsize * factor * scale
        self.meta["bpa"].val = deg2rad(pa)

    def get_xygrid(self, twodim=False, angunit=None):
        '''
        Get the xy coordinates of the image

        Args:
            angunit(string): Angular unit(uas, mas, asec or arcsec, amin or arcmin, degree)
            twodim(boolean): It True, the 2D grids will be returned. Otherwise, the 1D arrays will be returned

        Returns:
            x, y: x, y coordinates in the specified unit.
        '''
        from ..util.units import conv
        from numpy import meshgrid

        # get the conversion factor for the angular unit
        if angunit is None:
            angunit = self.angunit
        factor = conv("rad", angunit)

        x = self.data["x"].data * factor
        y = self.data["y"].data * factor
        if twodim:
            x, y = meshgrid(x, y)
        return x, y

    def get_extent(self, angunit=None):
        '''
        Get the field of view of the image for the pyplot.imshow function.

        Args:
          angunit(string): Angular unit

        Returns:
          [xmax, xmin, ymin, ymax]: extent of the image
        '''
        from ..util.units import conv
        from numpy import asarray

        # get the conversion factor for the angular unit
        if angunit is None:
            angunit = self.angunit
        factor = conv("rad", angunit)

        dx = self.meta["dx"].val
        dy = self.meta["dy"].val
        nx = self.meta["nx"].val
        ny = self.meta["ny"].val
        nxref = self.meta["nxref"].val
        nyref = self.meta["nyref"].val

        xmax = -dx * (0 - nxref - 0.5)
        xmin = -dx * (nx - 1 - nxref + 0.5)
        ymax = dy * (ny - 1 - nyref + 0.5)
        ymin = dy * (0 - nyref - 0.5)

        return asarray([xmax, xmin, ymin, ymax]) * factor

    # copy extent to imextent
    get_imextent = get_extent

    def get_beam(self, angunit=None):
        '''
        Get beam parameters.

        Args:
            angunit(string): Angular Unit
        Return:
            dic: the beam parameter information
        '''
        from ..util.units import conv
        from numpy import rad2deg

        if angunit is None:
            angunit = self.angunit
        factor = conv("rad", angunit)

        outdic = {}
        outdic["majsize"] = self.meta["bmaj"].val * factor
        outdic["minsize"] = self.meta["bmin"].val * factor
        outdic["pa"] = rad2deg(self.meta["bpa"].val)
        outdic["angunit"] = angunit
        return outdic

    def set_bconv(self, fluxunit="Jy", saunit="pixel"):
        from ..util.units import conv, Unit
        from numpy import log, pi

        # pixel size (in radian)
        dx = self.meta["dx"].val
        dy = self.meta["dy"].val
        dxdy = dx*dy  # solid angle of the pixel

        if "k" in fluxunit.lower():
            from astropy.constants import c, k_B
            nu = self.data.freq
            Jy2K = c.si.value ** 2 / \
                (2 * k_B.si.value * nu ** 2) / dxdy * 1e-26
            self.data["bconv"] = Jy2K * conv("K", fluxunit)
        else:
            fluxconv = conv("Jy", fluxunit)

            if "pix" in saunit.lower() or "px" in saunit.lower():
                saconv = 1
            elif "beam" in saunit.lower():
                # get beamsize (in radian)
                bmaj = self.meta["bmaj"].val
                bmin = self.meta["bmin"].val

                # beamsolid angle
                beamsa = bmaj*bmin*pi/(4*log(2))
                saconv = conv(dxdy*Unit("rad**2"), beamsa*Unit("rad**2"))
            else:
                saconv = conv(dxdy*Unit("rad**2"), saunit)

            self.data["bconv"] = fluxconv/saconv

    def get_imarray(self, fluxunit="Jy", saunit="pixel"):
        self.set_bconv(fluxunit=fluxunit, saunit=saunit)
        converted = self.data * self.data.bconv
        return converted.data.copy()

    # plotting function

    def imshow(self,
               scale="linear",
               dyrange=100,
               gamma=0.5,
               vmax=None,
               vmin=None,
               vscale="all",
               relative=False,
               fluxunit="Jy",
               saunit="pixel",
               restore=False,
               axisoff=False,
               axislabel=True,
               colorbar=False,
               colorbarprm={},
               cmap="afmhot",
               idx=(0, 0, 0),
               interpolation="bilinear",
               **imshow_args):
        '''
        Plot the image.
        To change the angular unit, please change IMFITS.angunit.

        Args:
          scale(str; default="linear"):
            Transfar function. Availables are "linear", "log", "gamma"
          dyrange(float; default=100):
            Dynamic range of the log color contour.
          gamma(float; default=1/2.):
            Gamma parameter for scale = "gamma".
          vmax(float):
            The maximum value of the color contour.
          vmin(float):
            The minimum value of the color contour.
            If logscale = True, dyrange will be used to set vmin.
          vscale(str):
            This specify how to set vmax and vmin
          relative(boolean, default=True):
            If True, vmin will be the relative value to the peak or vmax.
          fluxunit(string):
            Unit for the flux desity(Jy, mJy, uJy, K, si, cgs)
          saunit(string):
            Angular Unit for the solid angle(pixel, uas, mas, asec or arcsec,
            amin or arcmin, degree, beam). If restore is True, saunit will be
            forced to be "beam".
          axisoff(boolean, default=False):
            If True, plotting without any axis label, ticks, and lines.
            This option is superior to the axislabel option.
          axislabel(boolean, default=True):
            If True, plotting the axislabel.
          colorbar(boolean, default=False):
            If True, the colorbar will be shown.
          colorbarprm(dic, default={}):
            parameters for pyplot.colorbar
          index(integer):
          **imshow_args: Args will be input in matplotlib.pyplot.imshow
        '''
        from matplotlib.pyplot import imshow, axis
        from matplotlib.colors import LogNorm, PowerNorm
        from numpy import where, abs, isnan

        # Get angular unit
        angunit = self.angunit
        imextent = self.get_extent(angunit)

        # Get images to be plotted
        if len(idx) != 3:
            raise ValueError("len(idx) should be 3 [i_time, i_freq, i_stokes]")
        imarr = self.get_imarray(fluxunit=fluxunit, saunit=saunit)[idx]

        if vmax is None:
            peak = imarr.max()
        else:
            peak = vmax

        if scale.lower() == "log":
            vmin = None
            norm = LogNorm(vmin=peak/dyrange, vmax=peak)
            imarr[where(imarr < peak/dyrange)] = peak/dyrange
        elif scale.lower() == "gamma":
            if vmin is not None and relative:
                vmin *= peak
            elif vmin is None:
                vmin = 0.
            norm = PowerNorm(vmin=peak/dyrange, vmax=peak, gamma=gamma)
            imarr[where(abs(imarr) < 0)] = 0
        elif scale.lower() == "linear":
            if vmin is not None and relative:
                vmin *= peak
            norm = None
        else:
            raise ValueError(
                "Invalid scale parameters. Available: 'linear', 'log', 'gamma'")
        imarr[isnan(imarr)] = 0

        im = imshow(
            imarr, origin="lower", extent=imextent, vmin=vmin, vmax=vmax,
            cmap=cmap, interpolation=interpolation, norm=norm,
            **imshow_args
        )

        # Axis Label
        if axislabel:
            self.plot_xylabel()

        # Axis off
        if axisoff:
            axis("off")

        # colorbar
        '''
        if colorbar:
            clb = self.colorbar(fluxunit=fluxunit,
                                saunit=saunit, **colorbarprm)
            return im, clb
        else:
            return im
        '''

    def plot_xylabel(self):
        from ..util.plot import get_angunitlabel
        from matplotlib.pyplot import xlabel, ylabel

        angunit = self.angunit
        angunitlabel = get_angunitlabel(angunit)
        xlabel("Relative RA (%s)" % (angunitlabel))
        ylabel("Relative Dec (%s)" % (angunitlabel))

    def plot_beam(self, boxfc=None, boxec=None, beamfc=None, beamec="white",
                  lw=1., alpha=0.5, x0=0.05, y0=0.05, boxsize=1.5, zorder=None):
        '''
        Plot beam in the header.
        To change the angular unit, please change IMFITS.angunit.

        Args:
            x0, y0(float, default=0.05):
                leftmost, lowermost location of the box
                if relative = True, the value is on transAxes coordinates
            relative(boolean, default=True):
                If True, the relative coordinate to the current axis will be
                used to plot data
            boxsize(float, default=1.5):
                Relative size of the box to the major axis size.
            boxfc, boxec(color formatter):
                Face and edge colors of the box
            beamfc, beamec(color formatter):
                Face and edge colors of the beam
            lw(float, default=1): linewidth
            alpha(float, default=0.5): transparency parameter(0 < 1) for the face color
        '''
        from ..util.units import conv
        from ..util.plot import arrays_ellipse, arrays_box
        from matplotlib.pyplot import plot, fill, gca
        from numpy import max

        angunit = self.angunit
        angconv = conv("rad", angunit)

        majsize = self.meta["bmaj"].val * angconv
        minsize = self.meta["bmin"].val * angconv
        pa = self.meta["bpa"].val

        offset = max([majsize, minsize])/2*boxsize

        # get the current axes
        ax = gca()

        # center
        xedge, yedge = ax.transData.inverted().transform(ax.transAxes.transform((x0, y0)))
        xcen = xedge - offset
        ycen = yedge + offset

        # get ellipce shapes
        xe, ye = arrays_ellipse(xcen, ycen, majsize, minsize, pa)

        xb, yb = arrays_box(xcen, ycen, offset*2, offset*2)

        # plot
        if boxfc is not None:
            fill(xb, yb, fc=boxfc, alpha=alpha, zorder=zorder)
        if beamfc is not None:
            fill(xe, ye, fc=beamfc, alpha=alpha, zorder=zorder)
        plot(xe, ye, lw, color=beamec, zorder=zorder)
        if boxec is not None:
            plot(xb, yb, lw, color=boxec, zorder=zorder)

    def plot_scalebar(self, x, y, length, ha="center", color="white", lw=1, **plotargs):
        '''
        Plot a scale bar

        Args:
            x, y ( in the unit of the current plot):
                x, y coordinates of the scalebar
            length (in the unit of the current plot):
                length of the scale bar
            ha(str, default="center"):
                The horizontal alignment of the bar.
                Available options is ["center", "left", "right"]
            plotars:
                Arbital arguments for pyplot.plot.
        '''
        from numpy import abs
        from matplotlib.pyplot import plot

        if ha.lower() == "center":
            xmin = x-abs(length)/2
            xmax = x+abs(length)/2
        elif ha.lower() == "left":
            xmin = x - abs(length)
            xmax = x
        elif ha.lower() == "right":
            xmin = x
            xmax = x + abs(length)
        else:
            raise ValueError("ha must be center, left or right")

        # plot
        plot([xmax, xmin], [y, y], color=color, lw=lw, **plotargs)
