from smili2 import uvdata
from smili2.util.units import HOUR, MIN, DAY, SEC
from astropy.time import Time, TimeDelta

# uvd = uvdata.load_uvfits("/data-smili/J2203TSTCL5_SUBSP.uvfits", \
#                          "/data-smili/J2203TSTCL5_SUBSP.zarr")

uvd = uvdata.load_zarr("/data-smili/J2203TSTCL5_SUBSP.zarr")

timetab_scan = uvd.scan.segment_scans(tap=-1)

timetab_13sec = uvd.scan.segment_scans(tap=13)

def mjd2iso(mjd):
    tobj = Time(mjd, format='mjd', scale='utc')
    iso = tobj.iso         # Time in ISO format
    return iso
