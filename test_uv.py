import sys
import numpy as np
from xarray import Dataset
import smili2.uvdata as uv

# uvd= uv.load_uvfits('/home/benkev/ALMA/alma/alma.alma.cycle7.10_noise.uvfits',
#                '/home/benkev/ALMA/alma_zarr')

stokesid2name = { # Possible values of CRVAL3 in .uvfits headers
    "+1": "I",
    "+2": "Q",
    "+3": "U",
    "+4": "V",
    "-1": "RR",
    "-2": "LL",
    "-3": "RL",
    "-4": "LR",
    "-5": "XX",
    "-6": "YY",
    "-7": "XY",
    "-8": "YX",
}

in_uvfits = '/home/benkev/ALMA/3C273.coeff.fittp.uvfits'
out_zarr = '/home/benkev/ALMA/3C273_zarr'

pseudoI = False

uvd = uv.load_uvfits(in_uvfits, out_zarr)

vistab = uvd.vistab
ds = uvd.vistab.ds
vs = ds.vis

vsar = vs.data #.compute()   # Extract visibility ndarray
flag = vs.flag.data #.compute() 
sig = vs.sigma.data #.compute()

vt = uv.switch_polrepr(vistab, 'stokes')

raise SystemExit


shape1 = list(vsar.shape)
shape1[-1] = 4             # Extend pol axis to 4 to hold 4 Stokes parameters
# New visibility ndarray for Stokes parameters
vsar1 = np.zeros(shape1, dtype=complex)
flag1 = -np.ones(shape1, dtype=np.int32)   # Assume ALL data recoverable (f=-1) 
sig1 =  np.zeros(shape1, dtype=float)

lpol = list(vs.stokes.data) # List of pols like ['RR', 'LL']

if lpol == ['RR', 'LL']:
    rr = vsar[:,:,:,0]
    ll = vsar[:,:,:,1]
    vsar1[:,:,:,0] = 0.5*(rr + ll)         # I = 1/2 (RR + LL)
    vsar1[:,:,:,1] = 0.                    # Q = 1/2 (RL + LR)
    vsar1[:,:,:,2] = 0.                    # U = 1/2j(RL - LR)
    vsar1[:,:,:,3] = 0.5*(rr - ll)         # V = 1/2 (RR - LL)
    sig_norm = 0.5*np.sqrt(sig[:,:,:,0]**2 + sig[:,:,:,1]**2) #np.linalg.norm()
    sig1[:,:,:,0] = sig_norm
    sig1[:,:,:,3] = sig_norm
    #
    # If flag_rr == 1 and flag_ll == 1: flag1 = 1
    # If flag_rr == 0 or  flag_ll == 0: flag1 = 0
    # If sig_norm is NaN, Inf, or any of sigs is 0:    flag1 = 0
    # Otherwise flag1 = -1
    #
    truth_tbl1 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
    truth_tbl0 = np.logical_or(flag[:,:,:,0] == 0, flag[:,:,:,1] == 0)
    
    truth_tbls = np.logical_or(np.isnan(sig_norm), np.isinf(sig_norm))
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,0] == 0)
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,1] == 0)
    
    flag1[truth_tbl1,:] = 1
    flag1[truth_tbl0,:] = 0
    flag1[truth_tbls,:] = 0  # rr or ll has a bad sigma
    flag1[:,:,:,1] = 0       # Only I and V matter, Q and U do not.
    flag1[:,:,:,2] = 0       # Only I and V matter, Q and U do not.
    

elif lpol == ['XX', 'YY']:
    xx = vsar[:,:,:,0]
    yy = vsar[:,:,:,1]
    vsar1[:,:,:,0] =  0.5* (xx + yy)       # I = 1/2 (XX + YY)
    vsar1[:,:,:,1] =  0.5* (xx - yy)       # Q = 1/2 (XX - YY)
    vsar1[:,:,:,2] =  0.                   # U = 1/2 (XY + YX)
    vsar1[:,:,:,3] =  0.                   # V = 1/2j(XY - YX)
    sig_norm = 0.5*np.sqrt(sig[:,:,:,0]**2 + sig[:,:,:,1]**2) #np.linalg.norm()
    sig1[:,:,:,0] = sig_norm
    sig1[:,:,:,1] = sig_norm
    #
    # If flag_xx == 1 and flag_yy == 1: flag1 = 1
    # If flag_xx == 0 or  flag_yy == 0: flag1 = 0
    # If sig_norm is NaN, Inf, or any of sigs is 0:    flag1 = 0
    # Otherwise flag1 = -1
    #
    truth_tbl1 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
    truth_tbl0 = np.logical_and(flag[:,:,:,0] == 0, flag[:,:,:,1] == 0)
    
    truth_tbls = np.logical_or(np.isnan(sig_norm), np.isinf(sig_norm))
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,0] == 0)
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,1] == 0)

    flag1[truth_tbl1,:] = 1
    flag1[truth_tbl0,:] = 0
    flag1[truth_tbls,:] = 0  # xx or yy has a bad sigma
    flag1[:,:,:,2] = 0       # Only I and Q matter, U and V do not.
    flag1[:,:,:,3] = 0       # Only I and Q matter, U and V do not.
    

elif lpol == ['RR', 'LL', 'RL', 'LR']:
    rr = vsar[:,:,:,0]
    ll = vsar[:,:,:,1]
    rl = vsar[:,:,:,2]
    lr = vsar[:,:,:,3]
    vsar1[:,:,:,0] =  0.5* (rr + ll)       # I = 1/2 (RR + LL)
    vsar1[:,:,:,1] =  0.5* (rl + lr)       # Q = 1/2 (RL + LR)
    vsar1[:,:,:,2] = -0.5j*(rl - lr)       # U = 1/2j(RL - LR)
    vsar1[:,:,:,3] =  0.5* (rr - ll)       # V = 1/2 (RR - LL)

    sig_norm = 0.5*np.sqrt(sig[:,:,:,0]**2 + sig[:,:,:,1]**2 +
                           sig[:,:,:,2]**2 + sig[:,:,:,3]**2) #np.linalg.norm()
    sig1[:,:,:,:] = sig_norm
    #
    # If flag_rr == 1 & flag_ll == 1 & flag_rl == 1 & flag_lr == 1: flag1 = 1
    # If flag_rr == 0 | flag_ll == 0 | flag_rl == 0 | flag_lr == 0: flag1 = 0
    # If sig_norm is NaN, Inf, or any of sigs is 0:    flag1 = 0
    # Otherwise flag1 = -1
    #
    
    truth_tbl1 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
    truth_tbl1 = np.logical_and(truth_tbl1, flag[:,:,:,2] == 1)
    truth_tbl1 = np.logical_and(truth_tbl1, flag[:,:,:,3] == 1)
    
    truth_tbl0 = np.logical_or(flag[:,:,:,0] == 0, flag[:,:,:,1] == 0)
    truth_tbl0 = np.logical_or(truth_tbl0, flag[:,:,:,2] == 0)
    truth_tbl0 = np.logical_or(truth_tbl0, flag[:,:,:,3] == 0)
    
    truth_tbls = np.logical_or(np.isnan(sig_norm), np.isinf(sig_norm))
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,0] == 0)
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,1] == 0)
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,2] == 0)
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,3] == 0)
    
    flag1[truth_tbl1,:] = 1
    flag1[truth_tbl0,:] = 0
    flag1[truth_tbls,:] = 0  # rr or ll or rl or lr has a bad sigma


    

elif lpol == ['XX', 'YY', 'XY', 'YX']:
    xx = vsar[:,:,:,0]
    yy = vsar[:,:,:,1]
    xy = vsar[:,:,:,2]
    yx = vsar[:,:,:,3]
    vsar1[:,:,:,0] =  0.5* (xx + yy)       # I = 1/2 (XX + YY)
    vsar1[:,:,:,1] =  0.5* (xx - yy)       # Q = 1/2 (XX - YY)
    vsar1[:,:,:,2] =  0.5* (xy + yx)       # U = 1/2 (XY + YX)
    vsar1[:,:,:,3] = -0.5j*(xy - yx)       # V = 1/2j(XY - YX)

    sig_norm = 0.5*np.sqrt(sig[:,:,:,0]**2 + sig[:,:,:,1]**2 +
                           sig[:,:,:,2]**2 + sig[:,:,:,3]**2) #np.linalg.norm()
    sig1[:,:,:,:] = sig_norm
    #
    # If flag_xx == 1 & flag_yy == 1 & flag_xy == 1 & flag_yx == 1: flag1 = 1
    # If flag_xx == 0 | flag_yy == 0 | flag_xy == 0 | flag_yx == 0: flag1 = 0
    # If sig_norm is NaN, Inf, or any of sigs is 0:    flag1 = 0
    # Otherwise flag1 = -1
    #
    
    truth_tbl1 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
    truth_tbl1 = np.logical_and(truth_tbl1, flag[:,:,:,2] == 1)
    truth_tbl1 = np.logical_and(truth_tbl1, flag[:,:,:,3] == 1)
    
    truth_tbl0 = np.logical_or(flag[:,:,:,0] == 0, flag[:,:,:,1] == 0)
    truth_tbl0 = np.logical_or(truth_tbl0, flag[:,:,:,2] == 0)
    truth_tbl0 = np.logical_or(truth_tbl0, flag[:,:,:,3] == 0)
    
    truth_tbls = np.logical_or(np.isnan(sig_norm), np.isinf(sig_norm))
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,0] == 0)
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,1] == 0)
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,2] == 0)
    truth_tbls = np.logical_or(truth_tbls, sig[:,:,:,3] == 0)
    
    flag1[truth_tbl1] = 1
    flag1[truth_tbl0] = 0
    flag1[truth_tbls] = 0  # xx or yy or xy or yx has a bad sigma


#sys.exit(0)

#
# Create visibility table as an xarray.DataArray
#
ds1 = Dataset(
    data_vars=dict(
        vis=(["data", "spw", "ch", "stokes"], vsar1)
    ),
    coords=dict(
        mjd=("data", vs.mjd.data),  # .compute()),
        dmjd=("data", vs.dmjd.data),
        usec=("data", vs.usec.data),
        vsec=("data", vs.vsec.data),
        wsec=("data", vs.wsec.data),
        antid1=("data", vs.antid1.data),
        antid2=("data", vs.antid2.data),
        flag=(["data", "spw", "ch", "stokes"], flag1),
        sigma=(["data", "spw", "ch", "stokes"], sig1),
        stokes=(["stokes"], ['I', 'Q', 'U', 'V']),       # ?? change ????
    )
)

vt = uv.VisTable(ds=ds1.sortby(["mjd", "antid1", "antid2"]))

