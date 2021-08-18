#
# Read a uvfits file and create its flag array for checking how well
# the switch_polrepr() function works with the flags in smili2.
#
import numpy as np
import matplotlib.pyplot as pl
import astropy.io.fits as pf

#in_uvfits = '/data-smili/J2203TSTCL5_SUBSP.uvfits'
in_uvfits = '/home/benkev/ALMA/J2203TSTCL5_SUBSP.uvfits'

hdul = pf.open(in_uvfits)
hi = hdul.info(output=False)
Nhdu = len(hi)

for ihdu in range(Nhdu):
    print(hi[ihdu][1])
    hduname = hi[ihdu][1]
    if "PRIMARY" in hduname.upper():
        ghdu = hdul[ihdu]

#
# Read the visibilities from the uvfits PRIMARY hdu
#    uvfits's original dimension is [data,dec,ra,spw,ch,pol,vis]
#    vis: real, imaginary, weight.
#    Weights ≤ 0 indicate that the visibility measurement is flagged
#    and that the values may not be in any way meaningful.
#    first, we reorganize the array to [pol,spw,ch]
#
Ndata, Ndec, Nra, Nspw, Nch, Npol, visdim = ghdu.data.data.shape

dat = ghdu.data.data[:,0,0,0,:,:]
wei = dat[:,:,:,2]
sig = np.float64(np.power(abs(wei), -0.5))
flg = np.int32(np.sign(wei)) 

sig1 = np.copy(sig)
flg1 = np.copy(flg)

#
# Sieve out bad data
#
macheps = np.finfo(np.float64).eps
tt_good = np.logical_and(np.isfinite(sig), sig > macheps)
tt_bad = np.logical_not(tt_good)

sig[tt_bad] = 0.
flg[tt_bad] = 0


#
# The same, in smili2/uvdata/uvfits.py:
#
# # check sig
# idx = np.where(np.isinf(sig))
# sig1[idx] = 0
# flg1[idx] = 0

# idx = np.where(np.isnan(sig))
# sig1[idx] = 0
# flg1[idx] = 0

# idx = np.where(sig < np.finfo(np.float64).eps)
# sig1[idx] = 0
# flg1[idx] = 0

