import sys
import numpy as np
from xarray import Dataset
import smili2.uvdata as uv

# uvd= uv.load_uvfits('/home/benkev/ALMA/alma/alma.alma.cycle7.10_noise.uvfits',
#                '/home/benkev/ALMA/alma_zarr')

inuvfits = '/home/benkev/ALMA/3C273.coeff.fittp.uvfits'
outzarr = '/home/benkev/ALMA/3C273_zarr'

uvd = uv.load_uvfits(inuvfits, outzarr)

ds = uvd.vistab.ds
vs = ds.vis

vsar = vs.data.compute()   # Extract visibility ndarray
flag = vs.flag.data.compute() 
sig = vs.sigma.data.compute() 

shape1 = list(vsar.shape)
shape1[-1] = 4             # Extend pol axis to 4 to hold 4 Stokes parameters
# New visibility ndarray for Stokes parameters
vsar1 = np.zeros(shape1, dtype=complex)
flag1 = np.zeros(shape1, dtype=np.int32)
sig1 = np.zeros(shape1, dtype=float)

lpol = list(vs.stokes.data) # List of pols like ['RR', 'LL']

if lpol == ['RR', 'LL']:
    rr = vsar[:,:,:,0]
    ll = vsar[:,:,:,1]
    vsar1[:,:,:,0] = 0.5*(rr + ll)         # I = 1/2 (RR + LL)
    vsar1[:,:,:,1] = 0.                    # Q = 1/2 (RL + LR)
    vsar1[:,:,:,2] = 0.                    # U = 1/2j(RL - LR)
    vsar1[:,:,:,3] = 0.5*(rr - ll)         # V = 1/2 (RR - LL)
    sig1[:,:,:,0] = 0.5*np.sqrt(sig[:,:,:,0]**2 + sig[:,:,:,1]**2)
    sig1[:,:,:,3] = 0.5*np.sqrt(sig[:,:,:,0]**2 + sig[:,:,:,1]**2)  # ??????
    flag1[:,:,:,0] = np.copy(flag[:,:,:,0])
    flag1[:,:,:,3] = np.copy(flag[:,:,:,1])

elif lpol == ['RR', 'LL', 'RL', 'LR']:
    rr = vsar[:,:,:,0]
    ll = vsar[:,:,:,1]
    rl = vsar[:,:,:,2]
    lr = vsar[:,:,:,3]
    vsar1[:,:,:,0] =  0.5* (rr + ll)       # I = 1/2 (RR + LL)
    vsar1[:,:,:,1] =  0.5* (rl + lr)       # Q = 1/2 (RL + LR)
    vsar1[:,:,:,2] = -0.5j*(rl - lr)       # U = 1/2j(RL - LR)
    vsar1[:,:,:,3] =  0.5* (rr - ll)       # V = 1/2 (RR - LL)

elif lpol == ['XX', 'YY']:
    xx = vsar[:,:,:,0]
    yy = vsar[:,:,:,1]
    xy = vsar[:,:,:,2]
    yx = vsar[:,:,:,3]
    vsar1[:,:,:,0] =  0.5* (xx + yy)       # I = 1/2 (XX + YY)
    vsar1[:,:,:,1] =  0.5* (xx - yy)       # Q = 1/2 (XX - YY)
    vsar1[:,:,:,2] =  0.5* (xy + yx)       # U = 1/2 (XY + YX)
    vsar1[:,:,:,3] = -0.5j*(xy - yx)       # V = 1/2j(XY - YX)

elif lpol == ['XX', 'YY', 'XY', 'YX']:
    xx = vsar[:,:,:,0]
    yy = vsar[:,:,:,1]
    vsar1[:,:,:,0] =  0.5* (xx + yy)       # I = 1/2 (XX + YY)
    vsar1[:,:,:,1] =  0.5* (xx - yy)       # Q = 1/2 (XX - YY)
    vsar1[:,:,:,2] =  0.                   # U = 1/2 (XY + YX)
    vsar1[:,:,:,3] =  0.                   # V = 1/2j(XY - YX)

flag = vs.flag.data.compute()

#sys.exit(0)

#
# Create visibility table as an xarray.DataArray
#
ds1 = Dataset(
    data_vars=dict(
        vis=(["data", "if", "ch", "stokes"], vsar1)
    ),
    coords=dict(
        mjd=("data", vs.mjd.data.compute()),
        dmjd=("data", vs.dmjd.data.compute()),
        usec=("data", vs.usec.data.compute()),
        vsec=("data", vs.vsec.data.compute()),
        wsec=("data", vs.wsec.data.compute()),
        antid1=("data", vs.antid1.data.compute()),
        antid2=("data", vs.antid2.data.compute()),
        flag=(["data", "if", "ch", "stokes"], flag1),
        sigma=(["data", "if", "ch", "stokes"], sig1),
        stokes=(["stokes"], ['I', 'Q', 'U', 'V']),
    )
)

vt = uv.VisTable(ds=ds1.sortby(["mjd", "antid1", "antid2"]))

vt.to_zarr(outzarr)   # ValueError: variable 'vis' already exists with
                      # different dimension sizes:
                      # {existing_sizes} != {new_sizes}.
                      #to_zarr() only supports changing dimension sizes
                      # when explicitly appending, but append_dim=None.

