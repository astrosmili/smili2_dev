import numpy as np
import matplotlib.pyplot as pl
from smili2 import uvdata
from smili2.util.units import HOUR, MIN, DAY, SEC
from astropy.time import Time, TimeDelta

uvd = uvdata.load_zarr("/data-smili/J2203TSTCL5_SUBSP.zarr")
# uvd = uvdata.load_zarr("/home/benkev/ALMA/J2203TSTCL5_SUBSP.zarr")

spw = 0
ch = 14
pol = 'LL'
dt = 43.     # seconds

vt = uvd.vistab
mjd = vt.ds.mjd
Ndat = len(mjd)
tim = np.array(mjd)

tobj = Time(mjd, format='mjd', scale='utc')
tsec = tobj.unix         # Time in UNIX format (seconds since the Epoch)

vtds = vt.ds

#
# Create a unique baseline id, blid, and set it as a coord
#
vtds.coords["blid"] = vtds.antid1 * 1000 + vtds.antid2
grp_bl =  vtds.groupby("blid") 
lgrp_bl =list(grp_bl) 

ii = 0
for el_bl in grp_bl:
    blid = el_bl[0]
    #print(ii, blid)
    vtds_bl = el_bl[1]
    grp_scn = vtds_bl.groupby('scanid')

    for el_scn in grp_scn:
        scanid = el_scn[0]
        vtds_scn = el_scn[1]

    ii += 1
    if ii == 10: break
    

    

