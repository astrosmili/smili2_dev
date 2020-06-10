import os, sys
import numpy as np

encoding = sys.getdefaultencoding()

dtype_hdf5_header = [('name', 'a32'), ('dtype', 'a16'), ('value', 'a32'),
                     ('unit', 'a16'), ('descr', 'a64')]



#
# Definition of the metadata header record class with four fields,
# dtype, value, unit, and description
#
class HeadRec(object):
    
    def __init__(self, dtype=None, val=None, unit=None, descr=None):
        self.dtype = dtype
        self.val = val
        self.unit = unit
        self.descr = descr
        
    def __str__(self):
        sval = str(self.val)
        unit = self.unit
        descr = self.descr       
        #
        # For builtin types str, int and float, use 'str', 'int' and 'float'.
        # For the numpy types, add 'np.' prefix to get type names like
        # 'np.int64' or 'np.float64'.
        #
        dtyp = self.dtype
        if (dtyp is str) or (dtyp is int) or (dtyp is float):    
            stype = np.dtype(self.dtype).name  + ', ' # Builtin types
        else:
            stype = 'np.' + np.dtype(self.dtype).name + ', '
        if unit is not None:
            outstr = stype + ' ' + sval + ' (' + unit + ')'
        else:
            outstr = stype + ' ' + sval
        if descr is not None:
            outstr += ' # ' + descr
            
        return outstr
        


#
# Definition of the metadata dictionary class
#
class HeadDict(dict):
    
    def __str__(self):
        d = {str(key) : '(' + str(self.__dict__[key]) + ')' \
             for key in self.__dict__.keys()}
        return str(d)
    
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    

#
# Convert the ASCII text header (as in hdf5 file) into dictionary header
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
        header: dictionary with elements <keyword> : HeadRec(<dtype>, <value>).
    '''

    header = HeadDict()
    
    for headelem in strheader:
        nam = str(headelem[0], encoding)
        typ = str(headelem[1], encoding)
        val = str(headelem[2], encoding)
        dtyp = eval(typ)
        nel = len(headelem)
        unit =  str(headelem[3], encoding) if nel == 4 else None
        descr = str(headelem[4], encoding) if nel == 5 else None
        header[nam] = HeadRec(dtyp, dtyp(val), unit, descr)
        
    return header




#
# Convert the dictionary header into ASCII text header to save in hdf5 file
#
def conv_header_dict_to_ascii(header):
    '''
    Input:
        header: dictionary with elements <keyword> : HeadRec(<dtype>, <value>).
    Returns:
        hdf5_header: numpy structured array of 3-element records 
                <keyword> <dtype> <value>,
                each element being zero-terminated ASCII byte strings.
                Such arrays are seamlessly saved in HDF5 files as the HDF5
                native tables. 
                     
    
    '''
    n_header_elements = len(header.keys())
    hdf5_header = np.empty(n_header_elements, dtype=dtype_hdf5_header)
    ielem = 0
    for name, rec in header.items():
        byte_name = bytes(name, encoding)  # The ASCII keyword
        dtyp = rec.dtype
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
            
        byte_value = bytes(str(rec.val), encoding)
        byte_unit = bytes(str(rec.unit), encoding)
        byte_descr = bytes(str(rec.descr), encoding)
        hdf5_header[ielem] = byte_name, byte_dtype, byte_value, \
                             byte_unit, byte_descr
        ielem += 1
        
    return hdf5_header



#
# Definition of the image header as a dictionary with the elements
#  <keyword> : HeadRec(<dtype>, <value>),
# where HeadRec() is the header record class
#
def init_header():
    '''
    As a class member, the metadata header is a dictionary,
    each element of which is <keyword>: HeadRec(<dtype>, <value>).
    For example, to access RA, one should issue header['x'].val
    The dtype of RA is obtained as header['x'].dtype
    '''

    hdr = HeadDict()
    
    # Information
    hdr['object'] =     HeadRec(str, 'NONE', None, 'Astronomical object')
    hdr['telescope'] =  HeadRec(str, 'NONE', None, 'Telescope or array')
    hdr['instrument'] = HeadRec(str, 'NONE', None,
                                'Observation instrument')
    hdr['observer'] =   HeadRec(str, 'NONE', None, 'Institution or group')
    hdr['dateobs'] =    HeadRec(str, 'NONE', None, 'Date of observation')

    # RA information    
    hdr['x'] =     HeadRec(np.float64,  0., 'deg', 'Right Ascension')
    hdr['dx'] =    HeadRec(np.float64, -1., 'deg',
                           'Distance between pixels, latitudinal')
    hdr['nx'] =    HeadRec(np.int64,    1, 'pix', 'latitudinal image size')
    hdr['nxref'] = HeadRec(np.float64,  1.)

    # Dec information
    hdr['y'] =     HeadRec(np.float64, 0., 'deg', 'Declination')
    hdr['dy'] =    HeadRec(np.float64, 1., 'deg',
                           'Distance between pixels, longitudinal')
    hdr['ny'] =    HeadRec(np.int64,   1, 'pix', 'Longitudinal image size')
    hdr['nyref'] = HeadRec(np.float64, 1.)

    # Frequency (third axis) information
    hdr['f'] =     HeadRec(np.float64, 229.345e9, 'Hz', 'Frequency')
    hdr['df'] =    HeadRec(np.float64, 4e9, 'Hz', 'Frequency step')
    hdr['nf'] =    HeadRec(np.int64,   1, None, 'Number of frequencies')
    hdr['nfref'] = HeadRec(np.float64, 1.)

    # Stokes (fourth axis) Information
    hdr['s'] =     HeadRec(np.int64, 1, None, 'Polarization')
    hdr['ds'] =    HeadRec(np.int64, 1)
    hdr['ns'] =    HeadRec(np.int64, 1, None,
                           'Number of polarizations (1 to 4)')
    hdr['nsref'] = HeadRec(np.int64, 1)

    # Time (fifth axis) Information
    hdr['t'] =     HeadRec(np.float64, 0., 's', 'Time')
    hdr['dt'] =    HeadRec(np.float64, 1., 's', 'Integration time')
    hdr['nt'] =    HeadRec(np.int64,   1, None, 'Number of time frames')
    hdr['ntref'] = HeadRec(np.float64, 1.)

    # Beam information
    hdr['bmaj'] = HeadRec(np.float64, 0.)
    hdr['bin'] =  HeadRec(np.float64, 0.)
    hdr['bpa'] =  HeadRec(np.float64, 0.)

    return hdr



# def init_header():
#     '''
#     As a class member, the metadata header is a dictionary,
#     each element of which is <keyword>: HeadRec(<dtype>, <value>).
#     For example, to access RA, one should issue header['x'].val
#     The dtype of RA is obtained as header['x'].dtype
#     '''

#     header = {
#         # Information
#         'object':     hrec(str, 'NONE', None, 'Astronomical object'),
#         'telescope':  hrec(str, 'NONE', None, 'Telescope or array'),
#         'instrument': hrec(str, 'NONE', None, 'Observation instrument'),
#         'observer':   hrec(str, 'NONE', None, 'Institution or group'),
#         'dateobs':    hrec(str, 'NONE', None, 'Date of observation'),

#         # RA information    
#         'x':     hrec(np.float64,  0., 'deg', 'Right Ascension'),
#         'dx':    hrec(np.float64, -1., 'deg', 'Distance between pixels, ' \
#                       'latitudinal'),
#         'nx':    hrec(np.int64,    1, 'pix', 'latitudinal image size'),
#         'nxref': hrec(np.float64,  1.),

#         # Dec information
#         'y':     hrec(np.float64, 0., 'deg', 'Declination'),
#         'dy':    hrec(np.float64, 1., 'deg', 'Distance between pixels, ' \
#                       'longitudinal'),
#         'ny':    hrec(np.int64,   1, 'pix', 'Longitudinal image size'),
#         'nyref': hrec(np.float64, 1.),

#         # Frequency (third axis) information
#         'f':     hrec(np.float64, 229.345e9, 'Hz', 'Frequency'),
#         'df':    hrec(np.float64, 4e9, 'Hz', 'Frequency step'),
#         'nf':    hrec(np.int64,   1, None, 'Number of frequencies'),
#         'nfref': hrec(np.float64, 1.),

#         # Stokes (fourth axis) Information
#         's':     hrec(np.int64, 1, None, 'Polarization'),
#         'ds':    hrec(np.int64, 1),
#         'ns':    hrec(np.int64, 1, None, 'Number of polarizations (1 to 4)'),
#         'nsref': hrec(np.int64, 1),

#         # Time (fifth axis) Information
#         't':     hrec(np.float64, 0., 's', 'Time'),
#         'dt':    hrec(np.float64, 1., 's', 'Integration time'),
#         'nt':    hrec(np.int64,   1, None, 'Number of time frames'),
#         'ntref': hrec(np.float64, 1.),

#         # Beam information
#         'bmaj':  hrec(np.float64, 0.),
#         'bin':   hrec(np.float64, 0.),
#         'bpa':   hrec(np.float64, 0.)
#     }

#     return header



#
# Definition of the image header as a numpy structured array
# of 3-element records <keyword> <dtype> <value>,
# each element being zero-terminated ASCII byte strings.
# Such arrays are seamlessly saved in HDF5 files as the HDF5
# native tables. 
#
def init_header_ascii():
    '''
    Returns the image header as a numpy structured array
    of 3-element records <keyword> <dtype> <value>,
    each element being zero-terminated ASCII byte strings.

    This function is intended for testing purposes only. To initialise
    a header use the function
        init_header().
    The ascii header is created from the dictionary header with the function
        conv_header_dict_to_ascii(header)
    to be seamlessly saved in the HDF5 file as a native HDF5 table.
    '''
    
    hdf5_header = np.array([
        # Information
        ('object',     'str', 'None', 'None', 'Astronomical object'),
        ('telescope',  'str', 'None', b'None', b'Telescope or array'),
        ('instrument', 'str', 'None', b'None', b'Observation instrument'),
        ('observer',   'str', 'None', b'None', b'Institution or group'),
        ('dateobs',    'str', 'None', b'None', b'Date of observation'),

        # # RA information   
        # ('ra',    'np.float64', '0.'),
        # ('dx',    'np.float64', '-1.'),
        # ('nx',    'np.int64',   '1'),
        # ('nxref', 'np.float64', '1.'),

        # # Dec information
        # ('dec',   'np.float64', '0.'),
        # ('dy',    'np.float64', '1.'),
        # ('ny',    'np.int64',   '1'),
        # ('nyref', 'np.float64', '1.'),

        # # Third Axis Information
        # ('freq',  'np.float64', '229.345e9'),
        # ('dfreq', 'np.float64', '4e9'),
        # ('nf',    'np.int64',   '1'),
        # ('nfref', 'np.float64', '1.'),

        # # Stokes Information
        # ('s',     'np.int64', '1'),
        # ('ds',    'np.int64', '1'),
        # ('ns',    'np.int64', '1'),
        # ('nsref', 'np.int64', '1'),

        # # Beam information
        # ('bmaj', 'np.float64', '0.'),
        # ('bin',  'np.float64', '0.'),
        # ('bpa',  'np.float64', '0.'),
    ], dtype=dtype_hdf5_header)

    return hdf5_header










