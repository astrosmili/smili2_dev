import matplotlib.pyplot as plt
import smili2.imdata as imdata
from astropy.time import Time
import numpy as np
from alma_util import workdir

#
# Make this script running universally on our several machines.
#
basedir = workdir() + 'ALMA/'

fitsin =  basedir + 'RoundSpottyDisk2.fits'
fitsout = basedir + 'RoundSpottyDisk_smili.fits'

ttl = 'Round Spotty Disk. Freq = 46.1 GHz.'

cas = imdata.load_fits_casa(fitsin, imdtype=np.float32) 

cas.to_fits_casa(fitsout) 

#
# Frequency array
#
fr = np.copy(cas.data['freq'])
fr = 1e-9*fr

#
# MJD array to array of ISO times
#
# mjd = np.copy(cas.data['mjd'])
# tim = Time(mjd, format='mjd', scale='utc')
# iso = tim.iso

cas.imshow(colorbar=True, title=ttl) 

plt.show()


