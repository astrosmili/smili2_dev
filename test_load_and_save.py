import matplotlib.pyplot as plt
import smili2.imdata as imdata
from astropy.time import Time
import numpy as np

fitsin = '/home/benkev/SMILI/fits/Arp220-B5-spw1.image.pbcor.fits'
fitsout = '/home/benkev/SMILI/fits/Arp220_B5_spw1_smili2.fits'
ttl = 'Arp 220: Band 5, spw 1. Freq = %8.4f GHz.\n%s'

cas = imdata.load_fits_casa(fitsin) 

cas.to_fits_casa(fitsout) 

#
# Frequency array
#
fr = np.copy(cas.data['freq'])
fr = 1e-9*fr

#
# MJD array to array of ISO times
#
mjd = np.copy(cas.data['mjd'])
tim = Time(mjd, format='mjd', scale='utc')
iso = tim.iso

#cas.imshow(colorbar=True, interpolation='none', idx=(0,3,0)) 
cas.imshow(colorbar=True, title=ttl % (fr[0], iso[0]), idx=(0,3,0)) 
cas.imshow(colorbar=True, title=ttl % (fr[22], iso[0]), idx=(0,22,0)) 
cas.imshow(colorbar=True, title=ttl % (fr[29], iso[0]), idx=(0,29,0))
cas.imshow(colorbar=True, title=ttl % (fr[41], iso[0]), idx=(0,41,0)) 

plt.show()


