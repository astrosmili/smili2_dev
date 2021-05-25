import numpy as np
import smili2.uvdata as uv

# uvd= uv.load_uvfits('/home/benkev/ALMA/alma/alma.alma.cycle7.10_noise.uvfits',
#                '/home/benkev/ALMA/alma_zarr')

uvd = uv.load_uvfits('/home/benkev/ALMA/3C273.coeff.fittp.uvfits',
               '/home/benkev/ALMA/3C273_zarr')

ds = uvd.vistab.ds
vs = ds.vis

vsar = vs.data.compute()   # Extract visibility ndarray
shape1 = list(vsar.shape)
shape1[-1] = 4             # Extend pol axis to 4 to hold 4 Stokes parameters
vsar1 = np.zeros(shape1)   # New visibility ndarray for Stokes parameters

