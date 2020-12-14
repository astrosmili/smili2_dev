import matplotlib.pyplot as plt
import smili2.imdata as imdata
cas = imdata.load_fits_casa('/home/benkev/SMILI/fits/casa_example.fits') 

cas.to_fits_casa('/home/benkev/SMILI/fits/c.fits') 

cas.imshow() 
plt.show()


