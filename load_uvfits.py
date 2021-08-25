info = '''

load_uvfits.py
Opens a uvfits file to get access to its visibility (and other) data.

'''
import numpy as np
import astropy.io.fits as pf
import os

in_uvfits = '/home/benkev/ALMA/J2203TSTCL5_SUBSP.uvfits'

hdul = pf.open(in_uvfits)
hduinfo = hdul.info(output=False)
Nhdu = len(hduinfo)

fqtab = None
antab = None
ghdu = None

for ihdu in range(Nhdu):
    hduname = hduinfo[ihdu][1]
    if "PRIMARY" in hduname.upper():
        if ghdu is not None:
            warn("This UVFITS has more than two Primary HDUs.")
            warn("The later one will be taken.")
        ghdu = hdul[ihdu]
    elif "FQ" in hduname.upper():
        if fqtab is not None:
            warn("This UVFITS has more than two AIPS FQ tables.")
            warn("The later one will be taken.")
        fqtab = hdul[ihdu]
    elif "AN" in hduname.upper():
        if antab is not None:
            warn("This UVFITS has more than two AIPS AN tables.")
            warn("The later one will be taken.")
        antab = hdul[ihdu]

# Read visibilities
#    uvfits's original dimension is [data,dec,ra,if,ch,pol,complex]
#    first, we reorganize the array to [pol,spw,ch]
Ndata, Ndec, Nra, Nspw, Nch, Npol, Ncplx = ghdu.data.data.shape

dat = ghdu.data.data[:,0,0,0,:,:,0] + 1j*ghdu.data.data[:,0,0,0,:,:,1]
wei = ghdu.data.data[:,0,0,0,:,:,2]
sig = np.float64(np.power(abs(wei), -0.5))
flg = np.int32(np.sign(wei)) 

# Read Random Parameters
paridxes = [None for i in range(9)]
parnames = ghdu.data.parnames
Npar = len(parnames)
jd1 = np.zeros(Ndata)
jd2 = np.zeros(Ndata)
for i in range(Npar):
    parname = parnames[i]
    if "UU" in parname:
        paridxes[0] = i+1
        usec = np.float64(ghdu.data.par(i))
    if "VV" in parname:
        paridxes[1] = i+1
        vsec = np.float64(ghdu.data.par(i))
    if "WW" in parname:
        paridxes[2] = i+1
        wsec = np.float64(ghdu.data.par(i))
    if "DATE" in parname:
        if paridxes[3] is None:
            paridxes[3] = i+1
            jd1 = np.float64(ghdu.data.par(i))
        elif paridxes[4] is None:
            paridxes[4] = i+1
            jd2 = np.float64(ghdu.data.par(i))
        else:
            errmsg = "Random Parameters have too many 'DATE' columns."
            raise ValueError(errmsg)
    if "BASELINE" in parname:
        paridxes[5] = i+1
        bl = np.float64(ghdu.data.par(i))
    if "SOURCE" in parname:
        paridxes[6] = i+1
        srcid = np.int32(ghdu.data.par(i))
    if "INTTIM" in parname:
        paridxes[7] = i+1
        inttim = np.float64(ghdu.data.par(i))
    if "FREQSEL" in parname:
        paridxes[8] = i+1
        freqsel = np.int32(ghdu.data.par(i))






