import numpy as np
import matplotlib.pyplot as pl
from smili2 import uvdata
from smili2.util.units import HOUR, MIN, DAY, SEC
from astropy.time import Time, TimeDelta

# uvd = uvdata.load_uvfits("/data-smili/J2203TSTCL5_SUBSP.uvfits", \
#                          "/data-smili/J2203TSTCL5_SUBSP.zarr")

uvd = uvdata.load_zarr("/data-smili/J2203TSTCL5_SUBSP.zarr")

# uvd = uvdata.load_zarr("/home/benkev/ALMA/J2203TSTCL5_SUBSP.zarr")

timetab_scan = uvd.scan.segment_scans(tap=-1)

timetab_43sec = uvd.scan.segment_scans(tap=43)
t43 = np.array(timetab_43sec['mjd'])
Nt43 = len(t43)

vt = uvd.vistab
mjd = vt.ds.mjd
Ndat = len(mjd)
tim = np.array(mjd)

tim0 = tim[0]
tim = tim - tim0
t43 = t43 - tim0

tl = []
tchunk = []
j = 0
for i in range(Ndat):
    if tim[i] < t43[j]:
        tchunk.append(tim[i])
    else:
        tl.append(tchunk)
        tchunk = []
        j = j + 1
        if j == Nt43:
            tl.append(list(tim[i:]))
            break

lths = [len(t) for t in tl] # Lengths of tl sublists

xax = np.arange(Ndat)  # X axis for tim

#
# Find in x43ax the tim indices where tim is the closest to t43.
# Place the minimal absolute differences in t43min
#
x43ax = np.zeros(Nt43, dtype=int)
t43min = np.zeros(Nt43, dtype=float)

for i in range(Nt43):
    dtim = tim - t43[i]
    t43min[i] = min(abs(dtim))
    x43ax[i] = np.argmin(abs(dtim))


red =   '#ff0000'
green = '#00a000'
blue =  '#0000ff'

pl.close('all')

pl.figure()
pl.plot(xax, tim, color=blue)
pl.plot(x43ax, t43, '.', color=red)
pl.grid(1)
pl.xlabel('Data point number')
pl.ylabel('blue: mjd, red: timetable (relative units)')

pl.figure();
pl.plot(xax[:-1], np.diff(tim), '.', color=blue)
pl.plot(x43ax[:-1], np.diff(t43), '.', color=red)
pl.grid(1)
pl.xlabel('Data point number')
pl.ylabel('blue: diff(mjd), red: diff(timetable)')


pl.show()


def mjd2iso(mjd):
    tobj = Time(mjd, format='mjd', scale='utc')
    iso = tobj.iso         # Time in ISO format
    return iso
