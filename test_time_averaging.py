import sys
import numpy as np
import matplotlib.pyplot as pl
from astropy.time import Time, TimeDelta
from xarray import Dataset
import smili2.uvdata as uv
 
zarr =  '/data-smili/J2203TSTCL5_SUBSP.zarr'
#zarr =  '/home/benkev/ALMA/J2203TSTCL5_SUBSP.zarr'

zuv = uv.load_zarr(zarr)

vt = zuv.vistab
mjd = vt.ds.mjd
Ndat = len(mjd)

tobj = Time(mjd, format='mjd', scale='utc')
iso = tobj.iso         # Time in ISO format

mjd = np.array(mjd)

a1 = np.array(vt.ds.antid1) 
a2 = np.array(vt.ds.antid2) 

sc = np.array(vt.ds.scanid)

# Starts of the scans
scidx = np.nonzero(np.diff(vt.ds.scanid))[0] + 1
scidx = np.hstack((0, scidx)) # Add zero as the first scan start

#
# Let's consider for example a specific, 3rd scan
#

isc3 = scidx[3]
isc4 = scidx[4]

mjd3 = mjd[isc3:isc4]
iso3 = iso[isc3:isc4]
a13 = a1[isc3:isc4]
a23 = a2[isc3:isc4]

mjdidx = np.nonzero(np.diff(mjd))[0] + 1   # Where time changes
mjdidx = np.hstack((0, mjdidx)) # Add zero as the first mjd index

mjd3idx = isc3 + np.nonzero(np.diff(mjd3))[0] + 1
# Add isc3 as the mjd3 start and isc4 as mjd3 end
mjd3idx = np.hstack((isc3, mjd3idx, isc4-1))


#
# Visualize
#
Ndat = len(mjd)
tax = np.arange(Ndat)
tax3 = tax[isc3:isc4]  # Indices of scan 3

red =   '#ff0000'
green = '#00a000'
blue =  '#0000ff'

pl.close('all')

fig1 = pl.figure(figsize=(12,5))

ax1 = pl.subplot(121)
ax1.plot(tax, mjd-mjd[0], color=blue); ax1.grid(1); pl.title('mjd - mjd[0]')
ax1.plot(tax[isc3:isc4], mjd[isc3:isc4]-mjd[0], color=green, lw=2)

ax2 = pl.subplot(122)
ax2.plot(tax[isc3:isc4], mjd[isc3:isc4]-mjd[isc3], color=green); ax2.grid(1);
ax2.plot(tax[mjd3idx], mjd[mjd3idx]-mjd[isc3], 'r.')
pl.title('mjd[isc3:isc4] - mjd[isc3]')

fig2 = pl.figure()
pl.plot(tax, sc, color=blue); pl.plot(np.diff(vt.ds.scanid), color=red)
pl.grid(1)
pl.title('Scan ID and diff(vt.ds.scanid)')

pl.show()


