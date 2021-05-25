import smili2.uvdata as uv

# uvd= uv.load_uvfits('/home/benkev/ALMA/alma/alma.alma.cycle7.10_noise.uvfits',
#                '/home/benkev/ALMA/alma_zarr')

uvd = uv.load_uvfits('/home/benkev/ALMA/3C273.coeff.fittp.uvfits',
               '/home/benkev/ALMA/3C273_zarr')

ds = uvd.vistab.ds
vs = ds.vis

vst = uv.switch_polrepr('stokes')


