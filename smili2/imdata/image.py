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
                 nx=128, ny=None,
                 dx=2, dy=None,
                 ixref=None, iyref=None,
                 angunit="uas",
                 mjd=[0.],
                 freq=None,                               # [230e9],
                 nfreq=None, ifref=None, fref=None, fdel=None, funit=None,
                 ns=1,
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
            ixref ([type], optional): [description]. Defaults to None.
            iyref ([type], optional): [description]. Defaults to None.
            angunit (str, optional): [description]. Defaults to "uas".
            mjd (list, optional): [description]. Defaults to [0.].
            freq (list, optional): [description]. Defaults to [230e9].
            nfreq (int, optional),
            ifref (int, optional), 
            fref(float, optional), 
            fdel(float, optional)
            funit(str, optional): CRPIX, CRVAL, CDELT, and CUNIT for frequency.
            source (str): The source name.
            srccoord (astropy.coordinates.SkyCoord):
                Coordinates of the source. If not specified,
            instrument (str, optional): The name of instrument, telescope.

        Attributes:
            data (xarray.DataArray): Image data
            meta (util.metadata.MetaData): Meta data
            angunit (str): Angular Unit
        """
        from numpy import float64, int32, abs, isscalar, arange
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

        # ixref & iyref
        if ixref is None:
            args["ixref"] = args["nx"]/2 - 0.5
        else:
            args["ixref"] = float64(ixref)
        if iyref is None:
            args["iyref"] = args["ny"]/2 - 0.5
        else:
            args["iyref"] = float64(iyref)


        # nfreq, ifref, fref, fdel, funit - if present, freq param is ignored
        if nfreq or ifref or fref or fdel:
            # nfreq
            if nfreq is None:
                nfreq = int32(1)

            # ifref
            if ifref is None:
                args["ifref"] = int32(0)
            else:
                args["ifref"] = int32(ifref)

            # fref
            if fref is None:
                args["fref"] = float64(230e9)
            else:
                args["fref"] = float64(fref)

            # fdel
            if fdel is None:
                args["fdel"] = float64(1e9)
            else:
                args["fdel"] = float64(fdel)

            # funit
            if funit is None:
                args["funit"] = 'Hz      '
            else:
                args["funit"] = funit

            # freq = CRVAL + CDELT*(np.arange(NAXIS) - CRPIX + 1)
            freq = fref + fdel*(arange(nfreq) - ifref)
        else:
            # freq
            if freq is None:
                freq = [230e9]
            if isscalar(freq):
                args["freq"] = float64(freq)
            else:
                args["freq"] = float64(freq[0])


        # ns
        args["ns"] = int32(ns)

        # nt
        if isscalar(mjd):
            args["nt"] = 1
        else:
            args["nt"] = len(mjd)

        # nf
        if isscalar(freq):
            args["nf"] = 1
        else:
            args["nf"] = len(freq)

        #  angunit
        self.angunit = angunit

        # initialize meta data

        print('args = ', args)
        
        self._init_meta()
        for key in args.keys():
            if key not in self.meta.keys():
                raise ValueError("Key '{}' is not in the meta.".format(key))
            elif key in ["x", "y"]:
                raise ValueError("Use srccoord for the reference pixel " \
                                 "coordinates x and y.")
            self.meta[key].val = args[key]
            print('self.meta[key].val = args[', key, '] = ', args[key])
            

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
            ixref=MetaRec(val=0., unit="", dtype="float64",
                       comment="Pixel ID of the reference pixel in RA axis"),
            nx=MetaRec(val=1, unit="", dtype="int32",
                       comment="Number of pixels in RA axis"),
            # Dec
            y=MetaRec(val=0., unit="rad", dtype="float64",
                      comment="Dec of the reference pixel"),
            dy=MetaRec(val=1., unit="rad", dtype="float64",
                       comment="Pixel size in Dec direction"),
            iyref=MetaRec(val=0., unit="", dtype="float64",
                       comment="Pixel ID of the reference pixel in Dec axis"),
            ny=MetaRec(val=64, unit="", dtype="int32",
                       comment="Number of pixels in Dec axis"),
            # Stokes/Polarization
            ns=MetaRec(val=1, unit="", dtype="int32",
                       comment="Number of pixels in Stokes axis"),
            # Frequency
            freq=MetaRec(val=230e9, unit="Hz", dtype="float64",
                         comment="Frequency"),
            nf=MetaRec(val=1, unit="", dtype="int32",
                       comment="Number of pixels in Freq axis"),
            nfreq=MetaRec(val=1, unit="", dtype="int32",
                       comment="Number of pixels in Freq axis"),
            ifref=MetaRec(val=0, unit="", dtype="int32",
                          comment="Pixel ID of the reference pixel " \
                          "in Freq axis"),
            fref=MetaRec(val=230e9, unit="Hz", dtype="float64",
                         comment="Frequency at the reference pixel " \
                          "in Freq axis"),
            fdel=MetaRec(val=1e6, unit="Hz", dtype="float64",
                         comment="Frequency increment in Freq axis"),
            funit=MetaRec(val='Hz      ', unit="", dtype="float64",
                         comment="Frequency units from fits file header"),
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
        ns = self.meta["ns"].val

        if ns == 1:
            self.data["stokes"] = ["I"]
        elif ns == 4:
            self.data["stokes"] = ["I", "Q", "U", "V"]
        else:
            raise ValueError("Current version of SMILI accepts only" \
                             "single or full polarization images.")

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
            ixref = self.meta["i{}ref".format(axis)].val
            sign = sign_dx[axis]

            # compute coordinates
            self.data[axis] = sign*dx*(arange(nx)-ixref)

    def copy(self):
        from copy import deepcopy

        outimage = Image()
        outimage.meta = deepcopy(self.meta)
        outimage.data = deepcopy(self.data)
        outimage.angunit = deepcopy(self.angunit)
        return outimage

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

    def set_mjd(self, mjd, dmjd=None):
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

    def auto_angunit(self):
        from numpy import amax
        from ..util.units import Unit, DEG

        angunits = ["deg", "arcmin", "arcsec", "mas", "uas"]
        xmax = amax(self.get_imextent(angunit="deg"))*DEG

        for angunit in angunits:
            self.angunit = angunit
            if xmax < 0.1*Unit(angunit):
                continue
            else:
                break

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

    def get_imextent(self, angunit=None):
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
        ixref = self.meta["ixref"].val
        iyref = self.meta["iyref"].val

        xmax = -dx * (0 - ixref - 0.5)
        xmin = -dx * (nx - 1 - ixref + 0.5)
        ymax = dy * (ny - 1 - iyref + 0.5)
        ymin = dy * (0 - iyref - 0.5)

        return asarray([xmax, xmin, ymin, ymax]) * factor

    def get_imarray(self, fluxunit="Jy", saunit="pixel"):
        self.set_bconv(fluxunit=fluxunit, saunit=saunit)
        converted = self.data * self.data.bconv
        return converted.data.copy()

    def get_uvgrid(self, twodim=False):
        """
        Get the uv coordinates of the image on the Fourier domain

        Args:
            twodim(boolean): It True, the 2D grids will be returned. Otherwise, the 1D arrays will be returned

        Returns:
            u, v: u, v coordinates.
        """
        from numpy import meshgrid
        from numpy.fft import fftfreq, ifftshift

        # get the shape of array
        nt, nf, ns, ny, nx = self.data.shape
        del nt, nf, ns

        # pixel size
        dxrad = self.meta["dx"].val
        dyrad = self.meta["dy"].val

        # create uv grids
        ug = ifftshift(fftfreq(nx, d=-dxrad))
        vg = ifftshift(fftfreq(ny, d=dyrad))

        if twodim:
            return meshgrid(ug, vg)
        else:
            return ug, vg

    def get_uvextent(self):
        """
        Get the field of view of the image on the Fourier transform
        for the pyplot.imshow function. Here we assume that the Fourier image
        is created by the get_visarr method.

        Returns:
          [umax, umin, vmin, vmax]: extent of the Fourier transformed image.
        """
        from numpy import asarray
        ug, vg = self.get_uvgrid()
        du_half = (ug[1] - ug[0])/2
        dv_half = (vg[1] - vg[0])/2
        return asarray([ug[0]-du_half, ug[-1]+du_half, vg[0]-dv_half, vg[-1]+dv_half])

    def get_vis(self, idx=(0, 0, 0), apply_pulsefunc=True):
        """
        Get an array of visibilities computed from the image.

        Args:
            idx (tuple, optional):
                An index, or a list of indice of the image.
                The index should be in the form of (time, freq, stokes).
                Defaults to (0, 0, 0). If you specify None, then
                visibilities will be computed for every index of images.
            apply_pulsefunc (bool, optional):
                If True, the pulse function specified in the meta data
                will be applied. Defaults to True.

        Returns:
            numpy.ndarray:
                full complex visibilities. The array shape will
                depend on idx.
        """
        from numpy import pi, exp, asarray, unravel_index
        from numpy.fft import fftshift, ifftshift, fft2

        # get array
        imarr = self.get_imarray()
        nt, nf, ns, ny, nx = imarr.shape
        ixref = self.meta["ixref"].val
        iyref = self.meta["iyref"].val
        dxrad = self.meta["dx"].val
        dyrad = self.meta["dy"].val

        # get uv grid
        ug, vg = self.get_uvgrid(twodim=True)

        # adjust phase
        ix_cen = nx//2 + nx % 2  # the index of the image center used in np.fft.fft2
        iy_cen = ny//2 + ny % 2
        dix = ix_cen - ixref  # shift in the pixel unit
        diy = iy_cen - iyref
        viskernel = exp(1j*2*pi*(-dxrad*dix*ug + dyrad*diy*vg))

        # mutiply the pulse function
        if apply_pulsefunc:
            viskernel *= self.get_pulsefunc()(ug, vg)

        # define FFT function
        def dofft(imarr2d):
            return ifftshift(fft2(fftshift(imarr2d)))

        # compute full complex visibilities
        if idx is None:
            shape3d = (nt, nf, ns)
            vis = asarray([dofft(imarr[unravel_index(i, shape=shape3d)])
                           for i in range(nt*nf*ns)]).reshape([nt, nf, ns, ny, nx])
            vis[:, :, :] *= viskernel
        else:
            ndim = asarray(idx).ndim
            if ndim == 1:
                vis = dofft(imarr[idx])*viskernel
            elif ndim == 2:
                nidx = len(idx)
                vis = asarray([dofft(imarr[tuple(idx[i])])
                               for i in range(nidx)]).reshape([nidx, ny, nx])
                vis[:] *= viskernel
            else:
                raise ValueError("Invalid dimension of the input index.")
        return vis

    def get_source(self):
        """
        [summary]

        Returns:
            [type]: [description]
        """
        from astropy.coordinates import SkyCoord
        from ..util.units import RAD
        from ..source import Source

        name = self.meta["source"].val
        x = self.meta["x"].val * RAD
        y = self.meta["y"].val * RAD
        skycoord = SkyCoord(ra=x, dec=y)
        return Source(name=name, skycoord=skycoord)

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

    def get_pulsefunc(self):
        ptype = self.meta["ptype"].val.lower()

        if "delta" in ptype:
            def pulsefunc(u, v): return 1
        elif "rect" in ptype:
            from ..geomodel.models import Rectangular
            dxrad = self.meta["dx"].val
            dyrad = self.meta["dy"].val
            pulsefunc = Rectangular(
                Lx=dxrad, Ly=dyrad, dx=dxrad, dy=dyrad, angunit="rad").V
        else:
            raise ValueError("unknown pulse type: %s" % (ptype))
        return pulsefunc

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

    def add_geomodel(self, geomodel, idx=(0, 0, 0), inplace=False):
        """
        Add a specified geometric model to the image

        Args:
            geomodel (geomodel.GeoModel):
                The input geometric model.
            idx (tuble):
                An index of the image where the Gaussians to be added.
                Should be in the format of (time, freq, stokes).
                Default to (0,0,0).
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: if inplace==False.
        """
        if inplace:
            outimage = self
        else:
            outimage = self.copy()

        # get x,y coordinates
        xg, yg = self.get_xygrid(angunit="rad", twodim=True)

        # compute the intensity
        imarr = geomodel.I(xg, yg)

        # add data
        outimage.data[idx] += imarr

        if not inplace:
            return outimage

    def add_gauss(self, totalflux=1., x0=0., y0=0., majsize=1., minsize=None,
                  pa=0., scale=1., angunit="uas", inplace=False):
        """
        Add a Gaussian to the image.

        Args:
            totalflux (float, optional):
                total flux density in Jy.
            x0, y0 (float, optional):
                The centoroid position of the kernel.
            majsize, minsize (float, optional):
                The major- and minor- axis FWHM size of the kernel.
            pa (int, optional):
                The position angle of the kernel.
            scale (int, optional):
                The scaling factor to be applied to the kernel size.
            angunit (str, optional):
                The angular unit for the centroid location and kernel size.
                Defaults to "mas".
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: the convolved image if inplace==False.
        """
        from ..geomodel.models import Gaussian
        from ..util.units import conv

        # scale the axis size
        majsize_scaled = majsize * scale
        if minsize is None:
            minsize_scaled = majsize_scaled
        else:
            minsize_scaled = minsize * scale

        factor = conv("rad", angunit)
        dx = self.meta["dx"].val * factor
        dy = self.meta["dy"].val * factor

        # initialize Gaussian model
        geomodel = Gaussian(x0=x0, y0=y0, dx=dx, dy=dy, majsize=majsize_scaled,
                            minsize=minsize_scaled, pa=pa, angunit=angunit)

        # run convolution
        if inplace:
            self.add_geomodel(geomodel=geomodel, inplace=inplace)
        else:
            return self.add_geomodel(geomodel=geomodel, inplace=inplace)

    #  Convolution with Geometric Models
    def convolve_geomodel(self, geomodel, inplace=False):
        """
        Convolve the image with an input geometrical model

        Args:
            geomodel (geomodel.GeoModel):
                The input geometric model.
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: the convolved image if inplace==False.
        """
        from numpy import real, asarray, unravel_index, conj
        from numpy.fft import fftshift, ifftshift, fft2, ifft2

        if inplace:
            outimage = self
        else:
            outimage = self.copy()

        # get the array shape
        imarr = self.data.values.copy()
        nt, nf, ns, nx, ny = self.data.shape
        shape3d = (nt, nf, ns)

        # get uv coordinates and compute kernel
        # conj is applied because radio astronomy uses
        # "+" for the fourier exponent
        ug, vg = self.get_uvgrid(twodim=True)
        convkernel = conj(geomodel.V(ug, vg))

        # define convolve functions
        def dofft(imarr2d):
            return ifftshift(fft2(fftshift(imarr2d)))

        def doifft(vis2d):
            return real(ifftshift(ifft2(fftshift(vis2d))))

        def convolve2d(imarr2d):
            return doifft(dofft(imarr2d)*convkernel)

        # run fft convolve
        outimage.data.values = asarray([convolve2d(imarr[unravel_index(i, shape=shape3d)])
                                        for i in range(nt*nf*ns)]).reshape([nt, nf, ns, ny, nx])

        # return the output image
        if inplace is False:
            return outimage

    def convolve_gauss(self, x0=0., y0=0., majsize=1., minsize=None, pa=0., scale=1., angunit="uas", inplace=False):
        """
        Gaussian Convolution.

        Args:
            x0, y0 (float, optional):
                The centoroid position of the kernel.
            majsize, minsize (float, optional):
                The major- and minor- axis FWHM size of the kernel.
            pa (int, optional):
                The position angle of the kernel.
            scale (int, optional):
                The scaling factor to be applied to the kernel size.
            angunit (str, optional):
                The angular unit for the centroid location and kernel size.
                Defaults to "mas".
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: the convolved image if inplace==False.
        """
        from ..geomodel.models import Gaussian

        # scale the axis size
        majsize_scaled = majsize * scale
        if minsize is None:
            minsize_scaled = majsize_scaled
        else:
            minsize_scaled = minsize * scale

        # initialize Gaussian model
        geomodel = Gaussian(x0=x0, y0=y0, majsize=majsize_scaled,
                            minsize=minsize_scaled, pa=pa, angunit=angunit)

        # run convolution
        if inplace:
            self.convolve_geomodel(geomodel=geomodel, inplace=inplace)
        else:
            return self.convolve_geomodel(geomodel=geomodel, inplace=inplace)

    def convolve_rectangular(self, x0=0., y0=0., Lx=1., Ly=None, angunit="mas", inplace=False):
        """
        Convolution with a rectangular kernel

        Args:
            x0, y0 (float, optional):
                The centoroid position of the kernel.
            Lx, Ly (float, optional):
                The width of the kernel.
            angunit (str, optional):
                The angular unit for the centroid location and kernel size.
                Defaults to "mas".
            inplace (bool, optional):
                If False, return the convolved image. Defaults to False.

        Returns:
            imdata.Image: the convolved image if inplace==False.
        """
        from ..geomodel.models import Rectangular
        from ..util.units import conv

        # get the pixel size of the image
        rad2aunit = conv("rad", angunit)
        dx = self.meta["dx"].val * rad2aunit
        dy = self.meta["dy"].val * rad2aunit

        # initialize Gaussian model
        geomodel = Rectangular(x0=x0, y0=y0, Lx=Lx, Ly=Ly, dx=dx, dy=dy,
                               angunit=angunit)

        # run convolution
        if inplace:
            self.convolve_geomodel(geomodel=geomodel, inplace=inplace)
        else:
            return self.convolve_geomodel(geomodel=geomodel, inplace=inplace)

    #  Regrid images (in x,y direction)
    def regrid(self, template, preconv=True, order=1):
        """
        Regrid the image (only in x and y coordinates) using the grid defined
        in the input template image.

        Args:
            template (imdata.Image object): 
                A template image for the new image grid.
            preconv (bool, optional):
                If True and the new image grid interval is larger, the input 
                image will be blurred with the rectangular pulse function of 
                the new grid. Defaults to True.
            order (int, optional): 
                The order of the spline interpolation, which has to be in the 
                range 0-5.. Defaults to 1.

        Returns:
            imdata.Image object: the regridded image.
        """
        from numpy import arange, zeros, meshgrid, unravel_index, asarray
        from scipy.ndimage import map_coordinates

        # get image grid information
        dx0 = self.meta["dx"].val
        dy0 = self.meta["dy"].val
        ixr0 = self.meta["ixref"].val
        iyr0 = self.meta["iyref"].val
        ns0 = self.meta["ns"].val
        nf0 = self.meta["nf"].val
        nt0 = self.meta["nt"].val
        nimage = ns0*nf0*nt0

        dx1 = template.meta["dx"].val
        dy1 = template.meta["dy"].val
        nx1 = template.meta["nx"].val
        ny1 = template.meta["ny"].val
        ixr1 = template.meta["ixref"].val
        iyr1 = template.meta["iyref"].val

        # pre convolution, if we regrid the input image to a more rough grid.
        if (dx1 > dx0 or dy1 > dy0) and preconv:
            inputimage = self.convolve_rectangular(
                Lx=dx1,
                Ly=dy1,
                angunit="rad"
            )
        else:
            inputimage = self

        # Compute the coordinate transfer function
        coord = zeros([2, nx1 * ny1])
        xgrid = (arange(nx1) - ixr1) * dx1 / dx0 + ixr0
        ygrid = (arange(ny1) - iyr1) * dy1 / dy0 + iyr0
        x, y = meshgrid(xgrid, ygrid)
        coord[0, :] = y.flatten()
        coord[1, :] = x.flatten()

        # image to be output
        outimage = Image(
            nx=nx1,
            ny=ny1,
            dx=dx1,
            dy=dy1,
            ixref=ixr1,
            iyref=iyr1,
            angunit="rad",
            mjd=self.data["mjd"].data,
            freq=self.data["freq"].data,
            ns=ns0,
            source=self.meta["source"].val,
            srccoord=self.get_source().skycoord,
            instrument=self.meta["instrument"].val
        )
        outimage.auto_angunit()

        # Do interpolation from the input image to the new image
        def do_interpolate(i_image):
            imjd, ifreq, ipol = unravel_index(i_image, shape=(nt0, nf0, ns0))
            imarr = map_coordinates(
                inputimage.data[imjd, ifreq, ipol],
                coord,
                order=order,
                mode='constant', cval=0.0, prefilter=True
            )
            return imarr

        outimarr = asarray([do_interpolate(i_image)
                            for i_image in range(nimage)]).reshape(outimage.data.shape)
        outimarr *= dx1 * dy1 / dx0 / dy0
        outimage.data.data[:] = outimarr[:]

        return outimage

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
        imextent = self.get_imextent(angunit)

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

        return im

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

    # Simulation
    def observe(self, array, freq, utc, dutc=None, add_thermal_noise=True):
        from ..uvdata import UVData

        uvd = UVData()
        uvd.set_array(array)
        uvd.set_source(self.get_source())
        uvd.set_freq(freq)
        uvd.init_vis(utc=utc, dutc=dutc)
        uvd.calc_uvw()
        uvd.apply_ellimit()
        uvd.eval(self, inplace=True)
        uvd.calc_sigma()
        if add_thermal_noise:
            uvd.add_thermal_noise(inplace=True)
        return uvd


    
    #
    # File Loaders
    #
    
    #
    # LOAD FITS EHTIM
    #
    
    @classmethod
    def load_fits_ehtim(cls, infits):
        """
        Load a FITS Image in ehtim's format into an imdata.Image instance.

        Args:
            infits (str or astropy.io.fits.HDUList):
                input FITS filename or HDUList instance

        Returns:
            imdata.Image: loaded image]
        """
        import astropy.io.fits as pf
        from numpy import abs, deg2rad
        from astropy.coordinates import SkyCoord
        from ..util.units import DEG

        isfile = False
        if isinstance(infits, str):
            hdulist = pf.open(infits)
            isfile = True
        elif isinstance(infits, pf.HDUList):
            hdulist = infits.copy()

        # number of the Stokes Parameter
        ns = len(hdulist)

        # ra axis
        xdeg = hdulist[0].header["OBSRA"]
        nx = hdulist[0].header["NAXIS1"]
        dx = abs(deg2rad(hdulist[0].header["CDELT1"]))
        ixref = hdulist[0].header["CRPIX1"]-1

        # dec axis
        ydeg = hdulist[0].header["OBSDEC"]
        ny = hdulist[0].header["NAXIS2"]
        dy = abs(deg2rad(hdulist[0].header["CDELT2"]))
        iyref = hdulist[0].header["CRPIX2"]-1

        # stokes axis
        ns = len(hdulist)

        # time axis
        mjd = [hdulist[0].header["MJD"]]

        # frequency
        freq = [hdulist[0].header["FREQ"]]

        # source
        source = hdulist[0].header["OBJECT"]
        srccoord = SkyCoord(ra=xdeg*DEG, dec=ydeg*DEG)

        # telescope
        instrument = hdulist[0].header["TELESCOP"]

        outimage = cls(
            nx=nx, ny=ny,
            dx=dx, dy=dy, angunit="rad",
            ixref=ixref, iyref=iyref,
            mjd=mjd,
            freq=freq,
            ns=ns,
            source=source,
            srccoord=srccoord,
            instrument=instrument
        )

        for i in range(ns):
            outimage.data[0, 0, i] = hdulist[i].data.copy()

        if isfile:
            hdulist.close()

        # update angunit
        outimage.auto_angunit()

        return outimage


    #
    # LOAD FITS SMILI
    #
    
    # @classmethod
    # def load_fits_smili(cls, infits):


    # @classmethod
    # def load_fits_aips(cls, infits):


    
    #
    # LOAD FITS CASA
    #
    
    @classmethod
    def load_fits_casa(cls, infits):
        """
        Load a FITS Image in CASA's format into an imdata.Image instance.
        Args:
            infits (str or astropy.io.fits.HDUList):
                input FITS filename or HDUList instance
        Returns:
            imdata.Image: loaded image
        """
        import astropy.io.fits as pf
        from numpy import abs, deg2rad, arange
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        from ..util.units import DEG

        isfile = False
        if isinstance(infits, str):
            hdulist = pf.open(infits)
            isfile = True
        elif isinstance(infits, pf.HDUList):
            hdulist = infits.copy()

        hdu0 = hdulist[0]
        
        # for k, v in hdu0.header.items():
        #     pass
            
        # ra axis
        xdeg = hdu0.header["OBSRA"]
        nx = hdu0.header["NAXIS1"]
        dx = abs(deg2rad(hdu0.header["CDELT1"]))
        ixref = hdu0.header["CRPIX1"] - 1

        # dec axis
        ydeg = hdu0.header["OBSDEC"]
        ny = hdu0.header["NAXIS2"]
        dy = abs(deg2rad(hdu0.header["CDELT2"]))
        iyref = hdu0.header["CRPIX2"] - 1

        # time axis 
        isot = hdu0.header["DATE-OBS"]         # In the ISO time format
        tim = Time(isot, format='isot', scale='utc') # An astropy.time object
        mjd = [tim.mjd]                              # Modified Julian Date

        # frequency
        nfreq = hdu0.header["NAXIS3"]
        fref =  hdu0.header["CRVAL3"]
        fdel =  hdu0.header["CDELT3"]
        ifref = hdu0.header["CRPIX3"] - 1
        funit = hdu0.header["CUNIT3"]
        # freq = CRVAL3 + CDELT3*(np.arange(NAXIS3) - CRPIX3 + 1)
        # freq = fref + fdel*(arange(nfreq) - ifref)

        # stokes axis
        ns = hdu0.header["NAXIS4"]
        #sref =  hdu0.header["CRVAL4"]
        #sdel =  hdu0.header["CDELT4"]
        #isref = hdu0.header["CRPIX4"] - 1
        #sunit = hdu0.header["CUNIT4"]

        # source
        source = hdu0.header["OBJECT"]
        srccoord = SkyCoord(ra=xdeg*DEG, dec=ydeg*DEG)

        # telescope
        instrument = hdu0.header["TELESCOP"]

        outimage = cls(
            nx=nx, ny=ny,
            dx=dx, dy=dy, angunit="rad",
            ixref=ixref, iyref=iyref,
            mjd=mjd,
            freq=None,   # Array freq will be created in the cls.__init__()
            nf=nfreq, ifref=ifref, fref=fref, fdel=fdel, funit=funit,
            ns=ns,
            source=source,
            srccoord=srccoord,
            instrument=instrument
        )

        for i in range(ns):
            outimage.data[0, 0, i] = hdulist[i].data[0,0].copy()

        if isfile:
            hdulist.close()

        # update angunit
        outimage.auto_angunit()

        return outimage

    

    #
    # File Exporters
    #
    
    def to_fits_ehtim(self, outfits=None, overwrite=True, idx=(0, 0)):
        '''
        save the image(s) to the image FITS file or HDUList in the eht-imaging
        library's format

        Args:
            outfits (string; default is None):
                FITS file name. If not specified, then HDUList object will be
                returned.
            overwrite (boolean):
                It True, an existing file will be overwritten.
            idx (list):
                Index for (MJD, FREQ)
        Returns:
            HDUList object if outfits is None
        '''
        from astropy.io.fits import PrimaryHDU, ImageHDU, HDUList
        from ..util.units import conv

        # Get the number of stokes parameters
        ns = self.data.shape[2]

        # Get the Image Array
        if len(idx) != 2:
            raise ValueError(
                "idx must be a two dimensional index for (mjd, freq)")
        else:
            imarr = self.data.data[idx]
            imjd, ifreq = idx

        # Create HDUs
        #   Some conversion factor
        rad2deg = conv("rad", "deg")

        hdulist = []
        # current EHTIM format assumes each HDU / stokes parameter
        for ipol in range(ns):
            stokes = self.data["stokes"].data[ipol]

            if ipol == 0:
                hdu = PrimaryHDU(imarr[ipol])
            else:
                hdu = ImageHDU(imarr[ipol], name=stokes)

            # set header
            hdu.header.set("OBJECT", self.meta["source"].val)
            hdu.header.set("CTYPE1", "RA---SIN")
            hdu.header.set("CTYPE2", "DEC--SIN")
            hdu.header.set("CDELT1", -self.meta["dx"].val*rad2deg)
            hdu.header.set("CDELT2", self.meta["dy"].val*rad2deg)
            hdu.header.set("OBSRA", self.meta["x"].val*rad2deg)
            hdu.header.set("OBSDEC", self.meta["y"].val*rad2deg)
            hdu.header.set("FREQ", self.data["freq"].data[ifreq]) 
            hdu.header.set("CRPIX1", self.meta["ixref"].val+1)
            hdu.header.set("CRPIX2", self.meta["iyref"].val+1)
            hdu.header.set("MJD", self.data["mjd"].data[imjd])
            hdu.header.set("TELESCOP", self.meta["instrument"].val)
            hdu.header.set("BUNIT", "JY/PIXEL")
            hdu.header.set("STOKES", stokes)

            # appended to HDUList
            hdulist.append(hdu)

        # Convert the list of HDUs to HDUList
        hdulist = HDUList(hdulist)

        # return or write HDUList
        if outfits is None:
            return hdulist
        else:
            hdulist.writeto(outfits, overwrite=True)



    def to_fits_casa(self, outfits=None, overwrite=True, idx=(0, 0)):
        '''
        Save the image(s) to the image FITS file or HDUList in the CASA format
        Args:
            outfits (string; default is None):
                FITS file name. If not specified, then HDUList object will be
                returned.
            overwrite (boolean):
                It True, an existing file will be overwritten.
            idx (list):
                Index for (MJD, FREQ)
        Returns:
            HDUList object if outfits is None
        '''
        from astropy.io.fits import PrimaryHDU, ImageHDU, HDUList
        from ..util.units import conv

        # Get the number of stokes parameters
        ns = self.data.shape[2]

        # Get the Image Array
        if len(idx) != 2:
            raise ValueError(
                "idx must be a two dimensional index for (mjd, freq)")
        else:
            imarr = self.data.data[idx]
            imjd, ifreq = idx

        # Create HDUs
        #   Some conversion factor(s)
        rad2deg = conv("rad", "deg")

        hdulist = []

        hdu = PrimaryHDU(imarr[ipol])

        # set header
        hdu.header.set("OBJECT", self.meta["source"].val)

        hdu.header.set("CTYPE1", "RA---SIN")
        hdu.header.set("CRVAL1", self.meta["x"].val*rad2deg)
        hdu.header.set("CDELT1", -self.meta["dx"].val*rad2deg)
        hdu.header.set("CRPIX1", self.meta["ixref"].val+1)
        #hdu.header.set("CUNIT1", self.meta["x"].unit)
        hdu.header.set("CUNIT1", "deg")

        hdu.header.set("CTYPE2", "DEC--SIN")
        hdu.header.set("CRVAL2", self.meta["y"].val*rad2deg)
        hdu.header.set("CDELT2", self.meta["dy"].val*rad2deg)
        hdu.header.set("CRPIX2", self.meta["iyref"].val+1)
        #hdu.header.set("CUNIT2", self.meta["y"].unit)
        hdu.header.set("CUNIT2", "deg")

        hdu.header.set("CTYPE3", "FREQ    ")
        hdu.header.set("CRVAL3", self.meta["fref"].val)
        hdu.header.set("CDELT3", self.meta["fdel"].val)
        hdu.header.set("CRPIX3", self.meta["ifref"].val+1)
        hdu.header.set("CUNIT3", self.meta["funit"].val)

        hdu.header.set("CTYPE4", "STOKES  ")
        hdu.header.set("CRVAL4", 1.0)
        hdu.header.set("CDELT4", 1.0)
        hdu.header.set("CRPIX4", 1.0)
        hdu.header.set("CUNIT4", '        ')

        hdu.header.set("OBSRA", self.meta["x"].val*rad2deg)
        hdu.header.set("OBSDEC", self.meta["y"].val*rad2deg)
        hdu.header.set("FREQ", self.data["freq"].data[ifreq])
        hdu.header.set("MJD", self.data["mjd"].data[imjd])
        hdu.header.set("TELESCOP", self.meta["instrument"].val)
        hdu.header.set("BUNIT", "JY/PIXEL")
        hdu.header.set("STOKES", stokes)

        # appended to HDUList
        hdulist.append(hdu)

        # Convert the list of HDUs to HDUList
        hdulist = HDUList(hdulist)

        # return or write HDUList
        if outfits is None:
            return hdulist
        else:
            hdulist.writeto(outfits, overwrite=True)


# shortcut to I/O functions
load_fits_ehtim = Image.load_fits_ehtim
load_fits_casa =  Image.load_fits_casa


