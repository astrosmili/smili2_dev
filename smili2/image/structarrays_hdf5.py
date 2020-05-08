import numpy as np
import h5py 
import sys

#
# In Python3 the str type is the Unicode strings ("Unicode points sequences").
# There are several standard encodings, of which the most popular is 'utf-8'.
# The call below returns the default encoding (most likely, 'utf-8')
#
encoding = sys.getdefaultencoding()

dtype_hdf5_header = np.dtype([('name', 'a20'), ('dtype', 'a10'),
                              ('value', 'a32')])


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
        #print('headelem = "' + headelem + '"')
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


#
# Save structured arrays in an hdf5 file using h5py 
#

xx = np.array([('hue', 23), ('shade', -17), ('tint', 12)],
			  dtype=[('name', 'S10'), ('value', 'i4')])

header1 = np.array([ ('ra', b'17.761122472551133'),
					('dec', b'-29.00781055609385'),
					('mjd', b'57850'),
					('pol_prim', b'I'),
					('polrep', b'stokes'),
					('psize', b'4.848136811094e-12'),
					('rf', b'227070703125.0'),
					('source', b'SGRA')],
				  dtype=[('name', 'S20'), ('value', 'S32')])

header2 = np.array([ ('ra', b'', 17.761122472551133),
					 ('dec', b'', -29.00781055609385),
					 ('mjd', b'', 57850),
					 ('pol_prim', b'I', 0),
					 ('polrep', b'stokes', 0),
					 ('psize', b'4.848136811094e-12', 4.848136811094e-12),
					 ('rf', b'227070703125.0', 227070703125.0),
					 ('source', b'SGRA', 0)],
				   dtype=[('name', 'S20'), ('str', 'S32'), ('num', '<f8')])

aa = np.array([('hue', 23, 'int'), ('shade', -17, 'float'), \
               ('tint', 12, 'str')],
			  dtype=[('name', 'a10'), ('value', 'i4'), ('dtype', 'a10')])


#
# Definition of the image header as a Numpy structured array
#
def init_header_str():
    header = np.array([
        # Information
        ('object', 'str', 'NONE'),
        ('telescope', 'str', 'NONE'),
        ('instrument', 'str', 'NONE'),
        ('observer', 'str', 'NONE'),
        ('dateobs', 'str', 'NONE'),

        # RA information   
        ('ra', 'np.float64', '0.'),
        ('dx', 'np.float64', '-1.'),
        ('nx', 'np.int64', '1'),
        ('nxref', 'np.float64', '1.'),

        # Dec information
        ('dec', 'np.float64', '0.'),
        ('dy', 'np.float64', '1.'),
        ('ny', 'np.int64', '1'),
        ('nyref', 'np.float64', '1.'),

        # Third Axis Information
        ('freq', 'np.float64', '229.345e9'),
        ('dfreq', 'np.float64', '4e9'),
        ('nf', 'np.int64', '1'),
        ('nfref', 'np.float64', '1.'),

        # Stokes Information
        ('s', 'np.int64', '1'),
        ('ds', 'np.int64', '1'),
        ('ns', 'np.int64', '1'),
        ('nsref', 'np.int64', '1'),

        # Beam information
        ('bmaj', 'np.float64', '0.'),
        ('bin', 'np.float64', '0.'),
        ('bpa', 'np.float64', '0.'),
    ],
        dtype=dtype_hdf5_header)
    
    return header

#
# Definition of the image header as a Numpy structured array
#
def init_header():
    header = {
        # Information
        'object': [str, 'NONE'],
        'telescope': [str, 'NONE'],
        'instrument': [str, 'NONE'],
        'observer': [str, 'NONE'],
        'dateobs': [str, 'NONE'],

        # RA information   
        'ra': [np.float64, 0.],
        'dx': [np.float64, -1.],
        'nx': [np.int64, 1],
        'nxref': [np.float64, 1.],

        # Dec information
        'dec': [np.float64, 0.],
        'dy': [np.float64, 1.],
        'ny': [np.int64, 1],
        'nyref': [np.float64, 1.],

        # Third Axis Information
        'freq': [np.float64, 229.345e9],
        'dfreq': [np.float64, 4e9],
        'nf': [np.int64, 1],
        'nfref': [np.float64, 1.],

        # Stokes Information
        's': [np.int64, 1],
        'ds': [np.int64, 1],
        'ns': [np.int64, 1],
        'nsref': [np.int64, 1],

        # Beam information
        'bmaj': [np.float64, 0.],
        'bin': [np.float64, 0.],
        'bpa': [np.float64, 0.]
    }
    return header


strheader = init_header_str()

header_value = {}
header_dtype = {}

for headelem in strheader:
    #print('headelem = "' + headelem + '"')
    nam = str(headelem[0], encoding)
    typ = headelem[1]
    val =  headelem[2]
    dtyp = eval(typ)
    #print('nam=', nam, ', dtyp=', dtyp, ', val=', dtyp(val))
    header_value[nam] = dtyp(val)
    header_dtype[nam] = dtyp
    header[nam] = [dtyp, dtyp(val)]


#
# Convert the dictionary header into ASCII text header to save in hdf5 file
#
hdf5_header = conv_header_dict_to_ascii(header)

# n_header_elements = len(header.keys())
# hdf5_header = np.empty(n_header_elements, dtype=dtype_hdf5_header)
# iel = 0
# for name, dtyp_val in header.items():
#     byte_name = bytes(name, encoding)
#     dtyp = dtyp_val[0]
#     np_dtype = np.dtype(dtyp)
#     if (dtyp is str) or (dtyp is int) or (dtyp is float):    
#         byte_dtype = bytes(np_dtype.name, encoding)
#     else:
#         byte_dtype = bytes('np.' + np_dtype.name, encoding)
#     value = dtyp_val[1]
#     byte_value = bytes(str(value), encoding)
#     hdf5_header[iel] = byte_name, byte_dtype, byte_value
#     iel += 1
    

h5f = h5py.File('structarray_h5py.h5', 'w')
h5f.create_dataset('name_value', data=xx)
h5f.create_dataset('name_value_dtype', data=aa)
h5f.create_dataset('Metadata', data=header1)
h5f.create_dataset('Metadata2', data=header2)
h5f.create_dataset('Metadata_StrHeader', data=strheader)
h5f.close()


