import os, sys
import numpy as np

encoding = sys.getdefaultencoding()

dtype_hdf5_header = [('name', 'a20'), ('dtype', 'a10'), ('value', 'a32')]


#
# Definition of the metadata header record class with two fields, dtype and val
#
class hrec(object):
    
    def __init__(self, dtype=None, val=None, unit=None, descr=None):
        self.dtype = dtype
        self.val = val
        self.unit = unit
        self.descr = descr
        
    def __str__(self):
        stype = np.dtype(self.dtype).name
        sval = str(self.val)
        unit = self.unit
        descr = self.descr
        if unit == '':
            
        return '(' + stype + ' : ' + sval + ' (' + unit + '). ' + descr + ')'
        


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
        header: dictionary with elements <keyword> : hrec(<dtype>, <value>).
    '''

    header = {}
    
    for headelem in strheader:
        nam = str(headelem[0], encoding)
        typ = headelem[1]
        val =  headelem[2]
        dtyp = eval(typ)
        header[nam] = hrec(dtyp, dtyp(val))
        
    return header




#
# Convert the dictionary header into ASCII text header to save in hdf5 file
#
def conv_header_dict_to_ascii(header):
    '''
    Input:
        header: dictionary with elements <keyword> : hrec(<dtype>, <value>).
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
        dtyp = dtyp_val.dtype
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
        value = dtyp_val.val
        byte_value = bytes(str(value), encoding)
        hdf5_header[iel] = byte_name, byte_dtype, byte_value
        iel += 1
        
    return hdf5_header

#
# Definition of the image header as a dictionary with the elements
#  <keyword> : hrec(<dtype>, <value>),
# where hrec() is the header record class
#
def init_header():
    '''
    As a class member, the metadata header is a dictionary,
    each element of which is <keyword>: hrec(<dtype>, <value>).
    For example, to access RA, one should issue header['x'].val
    The dtype of RA is obtained as header['x'].dtype
    '''

    header = {
        # Information
        'object':     hrec(str, 'NONE'),
        'telescope':  hrec(str, 'NONE'),
        'instrument': hrec(str, 'NONE'),
        'observer':   hrec(str, 'NONE'),
        'dateobs':    hrec(str, 'NONE'),

        # RA information    
        'x':     hrec(np.float64,  0.),
        'dx':    hrec(np.float64, -1.),
        'nx':    hrec(np.int64,    1),
        'nxref': hrec(np.float64,  1.),

        # Dec information
        'y':     hrec(np.float64, 0.),
        'dy':    hrec(np.float64, 1.),
        'ny':    hrec(np.int64,   1),
        'nyref': hrec(np.float64, 1.),

        # Frequency hrec(third axis) information
        'f':     hrec(np.float64, 229.345e9),
        'df':    hrec(np.float64, 4e9),
        'nf':    hrec(np.int64,   1),
        'nfref': hrec(np.float64, 1.),

        # Stokes hrec(fourth axis) Information
        's':     hrec(np.int64, 1),
        'ds':    hrec(np.int64, 1),
        'ns':    hrec(np.int64, 1),
        'nsref': hrec(np.int64, 1),

        # Time hrec(fifth axis) Information
        't':     hrec(np.float64, 0.),
        'dt':    hrec(np.float64, 1.),
        'nt':    hrec(np.int64,   1),
        'ntref': hrec(np.float64, 1.),

        # Beam information
        'bmaj':  hrec(np.float64, 0.),
        'bin':   hrec(np.float64, 0.),
        'bpa':   hrec(np.float64, 0.)
    }

    return header



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
        ('object',     'str', 'NONE'),
        ('telescope',  'str', 'NONE'),
        ('instrument', 'str', 'NONE'),
        ('observer',   'str', 'NONE'),
        ('dateobs',    'str', 'NONE'),

        # RA information   
        ('ra',    'np.float64', '0.'),
        ('dx',    'np.float64', '-1.'),
        ('nx',    'np.int64',   '1'),
        ('nxref', 'np.float64', '1.'),

        # Dec information
        ('dec',   'np.float64', '0.'),
        ('dy',    'np.float64', '1.'),
        ('ny',    'np.int64',   '1'),
        ('nyref', 'np.float64', '1.'),

        # Third Axis Information
        ('freq',  'np.float64', '229.345e9'),
        ('dfreq', 'np.float64', '4e9'),
        ('nf',    'np.int64',   '1'),
        ('nfref', 'np.float64', '1.'),

        # Stokes Information
        ('s',     'np.int64', '1'),
        ('ds',    'np.int64', '1'),
        ('ns',    'np.int64', '1'),
        ('nsref', 'np.int64', '1'),

        # Beam information
        ('bmaj', 'np.float64', '0.'),
        ('bin',  'np.float64', '0.'),
        ('bpa',  'np.float64', '0.'),
    ], dtype=dtype_hdf5_header)

    return hdf5_header










