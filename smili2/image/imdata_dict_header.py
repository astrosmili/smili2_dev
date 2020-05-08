#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a sub-module of smili2 handling image data.
'''
__author__ = "Smili Developer Team"

import os, sys
import copy
import time
import numpy as np
import h5py

import astropy.coordinates as coord
import astropy.io.fits as pyfits
import astropy.time as at

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Internal
import util

#
# In Python3 the str type is the Unicode strings ("Unicode points
# sequences"). There are several standard encodings, of which the
# most popular is 'utf-8'.
# The call below returns the default encoding (most likely, 'utf-8')
#
encoding = sys.getdefaultencoding()

dtype_hdf5_header = [('name', 'a20'), ('dtype', 'a10'), ('value', 'a32')]


#
# Convert the dictionary header into ASCII text header to save in hdf5 file
#
def conv_header_ascii_to_dict(strheader):
    '''
    Input:
        strheader: numpy structured array of 3-element records 
                <keyword> <dtype> <value>,
                each element being zero-terminated ASCII byte strings.
                Such arrays are seamlessly saved in HDF5 files as the HDF5
                native tables. 
    Returns:
        header: dictionary with elements <keyword> :[<dtype>, <value>].
    '''

    header = {}
    
    for headelem in strheader:
        nam = str(headelem[0], encoding)
        typ = headelem[1]
        val =  headelem[2]
        dtyp = eval(typ)
        header_value[nam] = dtyp(val)
        header_dtype[nam] = dtyp
        header[nam] = [dtyp, dtyp(val)]
        
    return header




#
# Convert the ASCII text header (as in hdf5 file) into dictionary header
#
def conv_header_dict_to_ascii(header):
    '''
    Input:
        header: dictionary with elements <keyword> :[<dtype>, <value>].
    Returns:
        hdf5_header: numpy structured array of 3-element records 
                <keyword> <dtype> <value>,
                each element being zero-terminated ASCII byte strings.
                Such arrays are seamlessly saved in HDF5 files as the HDF5
                native tables. 
                     
    
    '''
    n_header_elements = len(header.keys())
    hdf5_header = np.empty(n_header_elements, dtype=dtype_hdf5_header)
    iel = 0
    for name, dtyp_val in header.items():
        byte_name = bytes(name, encoding)  # The ASCII keyword
        dtyp = dtyp_val[0]
        np_dtype = np.dtype(dtyp)
        #
        # For builtin types str, int and float, use 'str', 'int' and 'float'.
        # For the numpy types, add 'np.' prefix to get type names like
        # 'np.int64' or 'np.float64'.
        #
        if (dtyp is str) or (dtyp is int) or (dtyp is float):    
            byte_dtype = bytes(np_dtype.name, encoding) # Builtin types
        else:
            byte_dtype = bytes('np.' + np_dtype.name, encoding)
        value = dtyp_val[1]
        byte_value = bytes(str(value), encoding)
        hdf5_header[iel] = byte_name, byte_dtype, byte_value
        iel += 1
        
    return hdf5_header




#-------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------
class Image(object):
    '''
    Image class constructor
    '''
    def __init__(self, filename=None,
            imfitstype="standard",
            source=None,
            instrument=None,
            observer=None,
            dx=2., dy=None, angunit="uas",
            nx=100, ny=None, nxref=None, nyref=None,
            **args):

        self.imfitstype = imfitstype
        self.source = source
        self.instrument = instrument
        self.observer = observer
        self.angunit = angunit            # Default angular unit
        
        #
        # TimeStamp tstamp is a local time string
        #
        tstamp = time.strftime('%Y%b%d_%H%M%S', time.localtime())

        #
        # If filename is not specified, a new hdf5 file with the timestamp
        # name is created
        #
        if filename is None:
            filename = tstamp + '.hdf5'
            
        fname, fext = os.path.splitext(filename)
        fext = fext.lower()

        print(filename, fname, fext)
        
        if fext == '.fits':
            fits_fname = filename
            hdf5_fname = fname + '.hdf5'
        elif fext in ['.h5', '.hdf5']:
            hdf5_fname = filename
            fits_fname = fname + '.fits'

        self.hdf5_fname = hdf5_fname
        self.fits_fname = fits_fname
        self.imfits = imfits = fits_fname
        self.dtype_hdf5_header = dtype_hdf5_header

        #
        # Get conversion factor for angular scale
        #
        to_deg = util.angconv(angunit, "deg")

        #
        # Angular units conversion function
        #
        self.angconv = util.angconv  
        
        #
        # Get keys of Args
        #
        argkeys = list(args.keys())

        #
        # Create the metadata header as a dictionary with entries
        #     {<keyword>: [<dtype>, <value>], ...}
        #
        self.header = self.init_header()
        
        #
        # Image data
        #
        self.data = None

        #
        # Set Y pixel size
        #
        if dy is None:
            dy = np.abs(dx)
        self.header["dx"][1] = -np.abs(dx)
        self.header["dy"][1] =  dy
        self.dy = dy

        #
        # set X pixel size
        #
        if ny is None:
            ny = nx
        self.header["nx"][1] = nx
        self.header["ny"][1] = ny

        #
        # Reference pixel
        #
        if nxref is None:
            nxref = (nx+1.)/2
        if nyref is None:
            nyref = (ny+1.)/2
        self.header["nxref"][1] = nxref
        self.header["nyref"][1] = nyref
        
        #
        # Read header from keyword args
        #
        headerkeys = list(self.header.keys())
        for key in argkeys:
            if key in headerkeys:
                header_dtype = self.header[key][0]
                self.header[key][1] = header[key][0](args[key])

        self.header["x"][1]  *= to_deg
        self.header["y"][1]  *= to_deg
        self.header["dx"][1] *= to_deg
        self.header["dy"][1] *= to_deg
        
        self.data = np.zeros([self.header["nt"][1], self.header["ns"][1], \
                              self.header["nf"][1], \
                              self.header["ny"][1], self.header["nx"][1]], \
                             dtype=np.float64)


        #
        # After the header is ready, create the hdf5_header through conversion
        # from the dictionary header and save it
        # in the HDF5 file
        #
        #
        self.hdf5_header = hdf5_header = self.get_hdf5_header()
        self.hdf5_handle = h5py.File(hdf5_fname, 'w')
        self.hdf5_handle.create_dataset('Metadata', data=hdf5_header)
        
        # self.hdf5_handle.close()

        #
        # Initialize from the image fits file.
        #
        if imfits is not None:
            if  imfitstype=="standard":
                self.read_fits_standard(imfits)
            elif imfitstype=="ehtim":
                self.read_fits_ehtim(imfits)
            elif imfitstype=="aipscc":
                self.read_fits_aipscc(imfits)
            else:
                raise ValueError("imfitstype must be standard, ehtim or aipscc")

        # Set source, observer, instrument
        if instrument is not None:
            self.set_instrument(instrument)

        if observer is not None:
            self.set_observer(observer)

        if source is not None:
            self.set_source(source)
      
            
        #
        # Initialize fitsdata
        #
        #self.update_fits()


    #
    # Definition of the image header as a dictionary with elements
    # <keyword> :[<dtype>, <value>].
    #
    def init_header(self):
        '''
        As a class member, the metadata header is a dictionary,
        each element of which is <keyword>: [<dtype>, <value>].
        For example, to access RA, one should issue header['ra'][1].
        The dtype of RA is obtained as header['ra'][0].
        '''

        header = {
            # Information
            'object':     [str, 'NONE'],
            'telescope':  [str, 'NONE'],
            'instrument': [str, 'NONE'],
            'observer':   [str, 'NONE'],
            'dateobs':    [str, 'NONE'],

            # RA information    
            'x':     [np.float64,  0.],
            'dx':    [np.float64, -1.],
            'nx':    [np.int64,    1],
            'nxref': [np.float64,  1.],

            # Dec information
            'y':     [np.float64, 0.],
            'dy':    [np.float64, 1.],
            'ny':    [np.int64,   1],
            'nyref': [np.float64, 1.],

            # Frequency (third axis) information
            'f':     [np.float64, 229.345e9],
            'df':    [np.float64, 4e9],
            'nf':    [np.int64,   1],
            'nfref': [np.float64, 1.],

            # Stokes (fourth axis) Information
            's':     [np.int64, 1],
            'ds':    [np.int64, 1],
            'ns':    [np.int64, 1],
            'nsref': [np.int64, 1],

            # Time (fifth axis) Information
            't':     [np.float64, 0.],
            'dt':    [np.float64, 1.],
            'nt':    [np.int64,   1],
            'ntref': [np.float64, 1.],

            # Beam information
            'bmaj':  [np.float64, 0.],
            'bin':   [np.float64, 0.],
            'bpa':   [np.float64, 0.]
        }
        
        return header

   
    #
    # Convert the dictionary header into hdf5 ASCII text header
    #
    def get_hdf5_header(self):
        return conv_header_dict_to_ascii(self.header)

    
        
    #
    # Set source name and source coordinates
    #
    def set_source(self, source="SgrA*", srccoord=None):
        '''
        Set the source name and the source coordinate to the header.
        If source coordinate is not given, it will be taken from the CDS.

        Args:
            source (str; default="SgrA*"):
                Source Name
            srccoord (astropy.coordinates.Skycoord object; default=None):
                Source position. If not specified, it is automatically pulled
                from the CDS
        '''
        # get source coordinates if it is not given.
        if srccoord is None:
            srccoord = coord.SkyCoord.from_name(source)
        elif not isinstance(srccoord, coord.sky_coordinate.SkyCoord):
            raise ValueError("The source coordinate must be " \
                     "astropy.coordinates.sky_coordinate.SkyCoord obejct")

        #
        # Information
        #
        self.header["object"][1] = source
        self.header["x"][1] = srccoord.ra.deg
        self.header["y"][1] = srccoord.dec.deg
        self.update_fits()


        
    def set_instrument(self, instrument):
        '''
        Update headers for instrument and telescope with a
        specified name of the instrument.
        '''
        for key in ["instrument", "telescope"]:
            self.header[key][1] = self.header[key][0](instrument)


            
    def set_observer(self, observer):
        '''
        Update headers for instrument, telescope and observer with a
        specified name of the instrument.
        '''
        self.header["observer"][1] = self.header["observer"][0](observer)



    def set_beam(self, majsize=0., minsize=0., pa=0., scale=1., angunit=None):
        '''
        Set beam parameters into headers.

        Args:
            majsize, minsize (float, default=0):
                major/minor-axis FWHM size
            scale (float, default=1):
                scaling factor that will be multiplied to maj/min size.
            pa (float, default=0):
                position angle in deg

        '''
        if angunit is None:
            angunit = self.angunit
        to_deg = util.angconv(angunit, "deg")
        self.header["bmaj"][1] = majsize * to_deg * scale
        self.header["bmin"][1] = minsize * to_deg * scale
        self.header["bpa"][1] = pa



    def set_frequency(self, freq):
        '''
        Set the reference frequency into headers.

        Args:
            freq (float, default=0): the reference frequency in Hz.
        '''
        self.header["f"][1] = freq


    #
    # Read data from a standard image fits file
    #
    def read_fits_standard(self, imfits):
        '''
        Read data from the image FITS file.

        Args:
            imfits (string or hdulist, optional):
                If specified, image will be loaded from the specified 
                image fits file or specified HDUlist object.
        '''
        if isinstance(imfits, pyfits.hdu.hdulist.HDUList):
            hdulist = copy.deepcopy(imfits)
        else:
            hdulist = pyfits.open(imfits)
        self.hdulist = hdulist

        isx = False
        isy = False
        isf = False
        iss = False
        naxis = hdulist[0].header.get("NAXIS")
        for i in range(naxis):
            ctype = hdulist[0].header.get("CTYPE%d" % (i + 1))
            if ctype is None:
                continue
            if ctype[0:2] == "RA":
                isx = i + 1
            elif ctype[0:3] == "DEC":
                isy = i + 1
            elif ctype[0:4] == "FREQ":
                isf = i + 1
            elif ctype[0:6] == "STOKES":
                iss = i + 1

        if isx != False:
            self.header["nx"][1]    = hdulist[0].header.get("NAXIS%d" % (isx))
            self.header["x"][1]     = hdulist[0].header.get("CRVAL%d" % (isx))
            self.header["dx"][1]    = hdulist[0].header.get("CDELT%d" % (isx))
            self.header["nxref"][1] = hdulist[0].header.get("CRPIX%d" % (isx))
            for key in "nx,x,dx,nxref".split(","):
                self.header[key][1] = self.header[key][0](self.header[key][1])
        else:
            print("Warning: No image data along RA axis.")

        if isy != False:
            self.header["ny"][1]    = hdulist[0].header.get("NAXIS%d" % (isy))
            self.header["y"][1]     = hdulist[0].header.get("CRVAL%d" % (isy))
            self.header["dy"][1]    = hdulist[0].header.get("CDELT%d" % (isy))
            self.header["nyref"][1] = hdulist[0].header.get("CRPIX%d" % (isy))
            for key in "ny,y,dy,nyref".split(","):
                self.header[key][1] = self.header[key][0](self.header[key][1])
        else:
            print("Warning: No image data along DEC axis.")

        if isf != False:
            self.header["nf"]    = hdulist[0].header.get("NAXIS%d" % (isf))
            self.header["f"]     = hdulist[0].header.get("CRVAL%d" % (isf))
            self.header["df"]    = hdulist[0].header.get("CDELT%d" % (isf))
            self.header["nfref"] = hdulist[0].header.get("CRPIX%d" % (isf))
            for key in "nf,f,df,nfref".split(","):
                self.header[key][1] = self.header[key][0](self.header[key][1])
        else:
            print("Warning: No image data along FREQ axis.")

        if iss != False:
            self.header["ns"]    = hdulist[0].header.get("NAXIS%d" % (iss))
            self.header["s"]     = hdulist[0].header.get("CRVAL%d" % (iss))
            self.header["ds"]    = hdulist[0].header.get("CDELT%d" % (iss))
            self.header["nsref"] = hdulist[0].header.get("CRPIX%d" % (iss))
            for key in "ns,s,ds,nsref".split(","):
                self.header[key][1] = self.header[key][0](self.header[key][1])

        else:
            print("Warning: No image data along STOKES axis.")

        keys = "object,telescope,instrument,observer,dateobs".split(",")
        for key in keys:
            keyname = key.upper()[0:8]
            try:
                self.header[key] = hdulist[0].header.get(keyname)
                self.header_dtype[key](self.header[key])
            except:
                print("warning: FITS file doesn't have a header info of '%s'" \
                      %(keyname))

        # load data
        self.data = hdulist[0].data.reshape([self.header["ns"][1],
                                             self.header["nf"][1],
                                             self.header["ny"][1],
                                             self.header["nx"][1]])

        # get beam information
        try:
            bunit = hdulist[0].header.get("BUNIT").lower()
        except:
            bunit = "jy/pixel"

        if bunit=="jy/beam":
            keys = "bmaj,bmin,bpa".split(",")
            for key in keys:
                keyname = key.upper()
                try:
                    self.header[key] = hdulist[0].header.get(keyname)
                    self.header_dtype[key](self.header[key])
                except:
                    print("warning: FITS file doesn't have a header info of '%s'"%(keyname))
            self.data *= util.saconv(
                x1=self.header["bmaj"],
                y1=self.header["bmin"],
                angunit1="deg",
                satype1="beam",
                x2=self.header["dx"],
                y2=self.header["dy"],
                angunit2="deg",
                satype2="pixel",
            )

        self.update_fits()

        

    #
    # Read data from an AIPS image fits file
    #
    def read_fits_aipscc(self, imfits):
        '''
        Read data from the image FITS file. For the brightness distribution,
        this function loads the AIPS CC table rather than data of the primary 
        HDU.

        Args:
            imfits (string or hdulist, optional):
                If specified, image will be loaded from the specified image 
                fits file or specified HDUlist object.
        '''
        # Load FITS File
        self.read_fits_standard(imfits)
        self.header["nf"] = 1
        self.header["ns"] = 1
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        self.data = np.zeros([1,1,Ny,Nx])

        # Get AIPS CC Table
        aipscc = pyfits.open(imfits)["AIPS CC"]
        flux = aipscc.data["FLUX"]
        deltax = aipscc.data["DELTAX"]
        deltay = aipscc.data["DELTAY"]
        checkmtype = np.abs(np.unique(aipscc.data["TYPE OBJ"]))<1.0
        if False in checkmtype.tolist():
            raise ValueError("Input FITS file has non point-source CC " \
                             "components, which are not currently supported.")
        ix = np.int64(np.round(deltax/self.header["dx"] + \
                               self.header["nxref"] - 1))
        iy = np.int64(np.round(deltay/self.header["dy"] + \
                               self.header["nyref"] - 1))
        print("There are %d clean components in the AIPS CC Table."%(len(flux)))
        
        #
        # Add the brightness distribution to the image
        #
        count = 0
        for i in range(len(flux)):
            try:
                self.data[0,0,iy[i],ix[i]] += flux[i]
            except:
                count += 1
        if count > 0:
            print("%d components are ignore since they are outside " \
                  "of the image FoV."%(count))
        self.update_fits()




    #
    # Read data from the EHT image FITS file
    #
    def read_fits_ehtim(self, imfits):
        '''
        Read data from the image FITS file geneated from the EHT-imaging library

        Args:
            imfits (string or hdulist, optional):
                If specified, image will be loaded from the specified 
                image fits file or specified HDUlist object.
        '''
        import ehtim as eh
        im = eh.image.load_fits(imfits)
        obsdate = at.Time(im.mjd, format="mjd")
        obsdate = "%04d-%02d-%02d"%(obsdate.datetime.year, \
                                    obsdate.datetime.month, \
                                    obsdate.datetime.day)
        self.header["object"] = im.source
        self.header["x"] = im.ra * 12
        self.header["y"] = im.dec
        self.header["dx"] = -np.abs(im.psize * util.angconv("rad","deg"))
        self.header["dy"] = im.psize * util.angconv("rad","deg")
        self.header["nx"] = im.xdim
        self.header["ny"] = im.ydim
        self.header["nxref"] = im.xdim/2.+1
        self.header["nyref"] = im.ydim/2.+1
        self.header["f"] = im.rf
        self.header["dateobs"]=obsdate
        self.data = np.flipud(im.imvec.reshape([self.header["ny"], \
                                                self.header["nx"]]))
        self.data = self.data.reshape([1,1,self.header["ny"], \
                                       self.header["nx"]])
        self.update_fits()

        




    
    
    def update_fits(self,
                    bunit="JY/PIXEL",
                    cctab=True,
                    threshold=None,
                    relative=True,
                    istokes=0, ifreq=0):
        '''
        Reflect current self.data / self.header info to the image FITS data.
        Args:
            bunit (str; default=bunit): unit of the brightness. 
                ["JY/PIXEL", "JY/BEAM"] is available.
            cctab (boolean): If True, AIPS CC table is attached to fits file.
            istokes (integer): index for Stokes Parameter at which the image 
                will be used for CC table.
            ifreq (integer): index for Frequency at which the image will be 
                used for CC table.
            threshold (float): pixels with the absolute intensity smaller 
                than this value will be ignored in CC table.
            relative (boolean): If true, theshold value will be normalized 
                with the peak intensity of the image.
        '''

        # CREATE HDULIST
        hdu = pyfits.PrimaryHDU(self.data.copy())
        hdulist = pyfits.HDUList([hdu])

        # GET Current Time
        dtnow = dt.datetime.now()

        # FILL HEADER INFO
        hdulist[0].header.set("OBJECT",   self.header["object"])
        hdulist[0].header.set("TELESCOP", self.header["telescope"])
        hdulist[0].header.set("INSTRUME", self.header["instrument"])
        hdulist[0].header.set("OBSERVER", self.header["observer"])
        hdulist[0].header.set("DATE",     "%04d-%02d-%02d" %
                              (dtnow.year, dtnow.month, dtnow.day))
        hdulist[0].header.set("DATE-OBS", self.header["dateobs"])
        hdulist[0].header.set("DATE-MAP", "%04d-%02d-%02d" %
                              (dtnow.year, dtnow.month, dtnow.day))
        hdulist[0].header.set("BSCALE",   np.float64(1.))
        hdulist[0].header.set("BZERO",    np.float64(0.))
        if bunit.upper() == "JY/PIXEL":
            hdulist[0].header.set("BUNIT",    "JY/PIXEL")
            bconv = 1
        elif bunit.upper() == "JY/BEAM":
            hdulist[0].header.set("BUNIT",    "JY/BEAM")
            bconv = util.saconv(
                x1=self.header["dx"],
                y1=self.header["dy"],
                angunit1="deg",
                satype1="pixel",
                x2=self.header["bmaj"],
                y2=self.header["bmin"],
                angunit2="deg",
                satype2="beam",
            )
            hdulist[0].header.set("BMAJ",self.header["bmaj"])
            hdulist[0].header.set("BMIN",self.header["bmin"])
            hdulist[0].header.set("BPA",self.header["bpa"])
        hdulist[0].header.set("EQUINOX",  np.float64(2000.))
        hdulist[0].header.set("OBSRA",    np.float64(self.header["x"]))
        hdulist[0].header.set("OBSDEC",   np.float64(self.header["y"]))
        hdulist[0].header.set("DATAMAX",  self.data.max())
        hdulist[0].header.set("DATAMIN",  self.data.min())
        hdulist[0].header.set("CTYPE1",   "RA---SIN")
        hdulist[0].header.set("CRVAL1",   np.float64(self.header["x"]))
        hdulist[0].header.set("CDELT1",   np.float64(self.header["dx"]))
        hdulist[0].header.set("CRPIX1",   np.float64(self.header["nxref"]))
        hdulist[0].header.set("CROTA1",   np.float64(0.))
        hdulist[0].header.set("CTYPE2",   "DEC--SIN")
        hdulist[0].header.set("CRVAL2",   np.float64(self.header["y"]))
        hdulist[0].header.set("CDELT2",   np.float64(self.header["dy"]))
        hdulist[0].header.set("CRPIX2",   np.float64(self.header["nyref"]))
        hdulist[0].header.set("CROTA2",   np.float64(0.))
        hdulist[0].header.set("CTYPE3",   "FREQ")
        hdulist[0].header.set("CRVAL3",   np.float64(self.header["f"]))
        hdulist[0].header.set("CDELT3",   np.float64(self.header["df"]))
        hdulist[0].header.set("CRPIX3",   np.float64(self.header["nfref"]))
        hdulist[0].header.set("CROTA3",   np.float64(0.))
        hdulist[0].header.set("CTYPE4",   "STOKES")
        hdulist[0].header.set("CRVAL4",   np.int64(self.header["s"]))
        hdulist[0].header.set("CDELT4",   np.int64(self.header["ds"]))
        hdulist[0].header.set("CRPIX4",   np.int64(self.header["nsref"]))
        hdulist[0].header.set("CROTA4",   np.int64(0))

        # scale angunit
        hdulist[0].data *= bconv

        # Add AIPS CC Table
        if cctab:
            aipscctab = self.to_aipscc(threshold=threshold, relative=relative,
                    istokes=istokes, ifreq=ifreq)

            hdulist.append(hdu=aipscctab)

            next = len(hdulist)
            hdulist[next-1].name = 'AIPS CC'

        self.hdulist = hdulist



        
    def to_aipscc(self, threshold=None, relative=True,
                    istokes=0, ifreq=0):
        '''
        Make AIPS CC table

        Args:
            istokes (integer): index for Stokes Parameter at which the image 
                will be saved
            ifreq (integer): index for Frequency at which the image will be 
                saved
            threshold (float): pixels with the absolute intensity smaller 
                than this value will be ignored.
            relative (boolean): If true, theshold value will be normalized 
                with the peak intensity of the image.
        '''
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        xg, yg = self.get_xygrid(angunit="deg")
        X, Y = np.meshgrid(xg, yg)
        X = X.reshape(Nx * Ny)
        Y = Y.reshape(Nx * Ny)
        flux = self.data[istokes, ifreq]
        flux = flux.reshape(Nx * Ny)

        # threshold
        if threshold is None:
            thres = np.finfo(np.float64).eps
        else:
            if relative:
                thres = self.peak(istokes=istokes, ifreq=ifreq) * threshold
            else:
                thres = threshold
        thres = np.abs(thres)

        # adopt threshold
        X = X[flux >= thres]
        Y = Y[flux >= thres]
        flux = flux[flux >= thres]

        # make table columns
        c1 = pyfits.Column(name='FLUX', array=flux, format='1E',unit='JY')
        c2 = pyfits.Column(name='DELTAX', array=X, format='1E',unit='DEGREES')
        c3 = pyfits.Column(name='DELTAY', array=Y, format='1E',unit='DEGREES')
        c4 = pyfits.Column(name='MAJOR AX', array=np.zeros(len(flux)), \
                           format='1E',unit='DEGREES')
        c5 = pyfits.Column(name='MINOR AX', array=np.zeros(len(flux)), \
                           format='1E',unit='DEGREES')
        c6 = pyfits.Column(name='POSANGLE', array=np.zeros(len(flux)), \
                           format='1E',unit='DEGREES')
        c7 = pyfits.Column(name='TYPE OBJ', array=np.zeros(len(flux)), \
                           format='1E',unit='CODE')

        # make CC table
        tab = pyfits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7])
        return tab





    
    def to_difmapmod(self, outfile, threshold=None, relative=True,
                     istokes=0, ifreq=0):
        '''
        Save an image into a difmap model file

        Args:
          istokes (integer): index for Stokes Parameter at which the image 
              will be saved
          ifreq (integer): index for Frequency at which the image will be saved
          threshold (float): pixels with the absolute intensity smaller than 
              this value will be ignored.
          relative (boolean): If true, theshold value will be normalized with 
              the peak intensity of the image.
          save_totalflux (boolean): If true, the total flux of the image will 
              be conserved.
        '''
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        xg, yg = self.get_xygrid(angunit="mas")
        X, Y = np.meshgrid(xg, yg)
        R = np.sqrt(X * X + Y * Y)
        theta = np.rad2deg(np.arctan2(X, Y))
        flux = self.data[istokes, ifreq]

        R = R.reshape(Nx * Ny)
        theta = theta.reshape(Nx * Ny)
        flux = flux.reshape(Nx * Ny)

        if threshold is None:
            thres = np.finfo(np.float32).eps
        else:
            if relative:
                thres = self.peak(istokes=istokes, ifreq=ifreq) * threshold
            else:
                thres = threshold
        thres = np.abs(thres)

        f = open(outfile, "w")
        for i in np.arange(Nx * Ny):
            if np.abs(flux[i]) < thres:
                continue
            line = "%20e %20e %20e\n" % (flux[i], R[i], theta[i])
            f.write(line)
        f.close()



        

    def save_fits(self, outfitsfile=None, overwrite=True, bunit="Jy/pixel"):
        '''
        save the image(s) to the image FITS file or HDUList.

        Args:
            outfitsfile (string; default is None):
                FITS file name. If not specified, then HDUList object will be
                returned.
            overwrite (boolean):
                It True, an existing file will be overwritten.
        Returns:
            HDUList object if outfitsfile is None
        '''
        print("Warning: this method will be removed soon. " \
              "Please use the 'to_fits' method.")
        if outfitsfile is None:
            self.to_fits(outfitsfile, overwrite, bunit)
        else:
            return self.to_fits(outfitsfile, overwrite, bunit)






