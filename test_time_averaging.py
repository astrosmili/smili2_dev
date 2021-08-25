import sys
import numpy as np
import matplotlib.pyplot as pl
from astropy.time import Time, TimeDelta
from xarray import Dataset
import smili2.uvdata as uv
 
zarr =  '/home/benkev/ALMA/J2203TSTCL5_SUBSP_zarr'

zuv = uv.load_zarr(zarr)

vt = zuv.vistab
mjd = vt.ds.mjd

tobj = Time(mjd, format='mjd', scale='utc')
iso = tobj.iso 

a1 = np.array(vt.ds.antid1) 
a2 = np.array(vt.ds.antid2) 

sc = np.array(vt.ds.scanid)

#
# Visualize
#
fig1 = pl.figure()
pl.plot(mjd-mjd[0]); pl.grid(1); pl.title('mjd - mjd[0]')

fig2 = pl.figure()
pl.plot(sc); pl.plot(np.diff(vt.ds.scanid)); pl.grid(1)
pl.title('Scan ID and diff(vt.ds.scanid)')

# Starts of the scans
scidx = np.nonzero(np.diff(vt.ds.scanid))[0] + 1
scidx = np.hstack((0, scidx)) # Add zero as the first scan start

#
# Let's consider for example a specific, 3rd scan
#
isc0 = scidx[3]
isc1 = scidx[4]

fig3 = pl.figure()
pl.plot(mjd[isc0:isc1]-mjd[isc0]); pl.grid(1);
pl.title('mjd[isc0:isc1] - mjd[isc0]')

pl.show()
