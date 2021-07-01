import sys
import numpy as np
from xarray import Dataset
import smili2.uvdata as uv

# uvd= uv.load_uvfits('/home/benkev/ALMA/alma/alma.alma.cycle7.10_noise.uvfits',
#                '/home/benkev/ALMA/alma_zarr')

polid2name = { # Possible values of CRVAL3 in .uvfits headers
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

# raise SystemExit

vt = uv.switch_polrepr(vistab, "stokes")

