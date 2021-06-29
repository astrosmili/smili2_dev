#!/usr/bin/env python
# -*- coding: utf-8 -*-

def switch_polrepr(vistab, polrepr, pseudoI=False):
    """
    This function changes representation of the polarization of
    the visibility data provided in vistab (of class VisTable).
    It returns a new instance of class VisTable where the polarization
    is rendered according to the one provided in the polrepr string.
    
    Args:
    
    vistab  (smili2.uvdata.uvdata.vis.VisTable): uvfits visibility
                                                 data in smili2 format.
    polrepr (str): polarization representation, "stokes" or "circ" or "linear".
    pseudoI (bool): if True, calculate I from XX or YY or RR or LL. 
    
    """
    import numpy as np
    from xarray import Dataset
    from .vistab import VisTable

    vsar = vistab.ds.vis.data       #.compute()   # Extract visibility ndarray
    flag = vistab.ds.vis.flag.data  #.compute() 
    sig =  vistab.ds.vis.sigma.data.compute() 

    shape1 = list(vsar.shape)
    shape1[-1] = 4          # Extend pol axis to 4 to hold 4 Stokes parameters
    
    # New visibility ndarray for Stokes parameters
    vsar1 = np.zeros(shape1, dtype=complex)
    flag1 = np.zeros(shape1, dtype=np.int32) # Assume ALL the data are bad
    sig1 =  np.zeros(shape1, dtype=float)

    lpol = list(vistab.ds.vis.pol.data) # List of pols like ['RR', 'LL']

    if lpol == ['RR'] or lpol == ['LL'] or lpol == ['XX'] or lpol == ['YY']:
        
        vsar1[:,:,:,0] = np.copy(vsar[:,:,:,0])    # I ~ RR or LL or XX or YY
        vsar1[:,:,:,1:] = 0.
        sig1[:,:,:,0] = np.copy(sig[:,:,:,0])
        sig1[:,:,:,1:] = 0.
        flag1[:,:,:,0] = np.copy(flag[:,:,:,0])
        #flag1[:,:,:,1:] = 0.


    elif lpol == ['RR', 'LL']:
        rr = vsar[:,:,:,0]
        ll = vsar[:,:,:,1]

        # Find where both RR and LL are good: tt_ is "truth table"
        # "Finite" means "not Inf and not NaN"

        tt_all = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,0]))
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,1]))
        tt_all = np.logical_and(tt_all, sig[:,:,:,0] != 0)
        tt_all = np.logical_and(tt_all, sig[:,:,:,1] != 0)

        vsar1[tt_all,0] = 0.5*(rr[tt_all] + ll[tt_all])    # I = 1/2 (RR + LL)
        vsar1[tt_all,1] = 0.                               # Q = 1/2 (RL + LR)
        vsar1[tt_all,2] = 0.                               # U = 1/2j(RL - LR)
        vsar1[tt_all,3] = 0.5*(rr[tt_all] - ll[tt_all])    # V = 1/2 (RR - LL)

        sig_norm = 0.5*np.sqrt(sig[tt_all,0]**2 + sig[tt_all,1]**2)
        sig1[tt_all,0] = sig_norm
        sig1[tt_all,3] = sig_norm

        flag1[tt_all,0] = 1
        flag1[tt_all,3] = 1

        if pseudoI:
            # Find where either RR or LL are good: tt_ is "truth table"
            # "Finite" means "not Inf and not NaN"

            tt_rr = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] != 1)
            tt_rr = np.logical_and(tt_rr, np.isfinite(sig[:,:,:,0]))
            tt_rr = np.logical_and(tt_rr, sig[:,:,:,0] != 0)
            vsar1[tt_rr,0] = rr[tt_rr]             # I = RR
            vsar1[tt_rr,3] = 0.                    # V = 0

            tt_ll = np.logical_and(flag[:,:,:,0] != 1, flag[:,:,:,1] == 1)
            tt_ll = np.logical_and(tt_ll, np.isfinite(sig[:,:,:,1]))
            tt_ll = np.logical_and(tt_ll, sig[:,:,:,1] != 0)
            vsar1[tt_ll,0] = ll[tt_ll]             # I = LL
            vsar1[tt_ll,3] = 0.                    # V = 0

            flag1[tt_rr,0] = 1
            flag1[tt_ll,0] = 1

            
    elif lpol == ['XX', 'YY']:

        xx = vsar[:,:,:,0]
        yy = vsar[:,:,:,1]
        
        # Find where both XX and YY are good: tt_ is "truth table"
        # "Finite" means "not Inf and not NaN"

        tt_all = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,0]))
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,1]))
        tt_all = np.logical_and(tt_all, sig[:,:,:,0] != 0)
        tt_all = np.logical_and(tt_all, sig[:,:,:,1] != 0)

        vsar1[tt_all,0] = 0.5*(xx[tt_all] + yy[tt_all])    # I = 1/2 (XX + YY)
        vsar1[tt_all,1] = 0.5*(xx[tt_all] - yy[tt_all])    # Q = 1/2 (RL + LR)
        vsar1[tt_all,2] = 0.                               # U = 1/2j(RL - LR)
        vsar1[tt_all,3] = 0.                               # V = 1/2 (XX - YY)

        sig_norm = 0.5*np.sqrt(sig[tt_all,0]**2 + sig[tt_all,1]**2)
        sig1[tt_all,0] = sig_norm
        sig1[tt_all,1] = sig_norm

        flag1[tt_all,0] = 1
        flag1[tt_all,1] = 1

        if pseudoI:
            # Find where either XX or YY are good: tt_ is "truth table"
            # "Finite" means "not Inf and not NaN"

            tt_xx = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] != 1)
            tt_xx = np.logical_and(tt_xx, np.isfinite(sig[:,:,:,0]))
            tt_xx = np.logical_and(tt_xx, sig[:,:,:,0] != 0)
            vsar1[tt_xx,0] = xx[tt_xx]             # I = XX
            vsar1[tt_xx,3] = 0.                    # V = 0

            tt_yy = np.logical_and(flag[:,:,:,0] != 1, flag[:,:,:,1] == 1)
            tt_yy = np.logical_and(tt_yy, np.isfinite(sig[:,:,:,1]))
            tt_yy = np.logical_and(tt_yy, sig[:,:,:,1] != 0)
            vsar1[tt_yy,0] = yy[tt_yy]             # I = YY
            vsar1[tt_yy,3] = 0.                    # V = 0

            flag1[tt_xx,0] = 1
            flag1[tt_yy,0] = 1


    elif lpol == ['RR', 'LL', 'RL', 'LR']:
        
        rr = vsar[:,:,:,0]
        ll = vsar[:,:,:,1]
        rl = vsar[:,:,:,2]
        lr = vsar[:,:,:,3]

        # Find where RR, LL, RL, and LR are good: tt_ is "truth table"
        # "Finite" means "not Inf and not NaN"

        tt_all = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
        tt_all = np.logical_and(tt_all, flag[:,:,:,2] == 1)
        tt_all = np.logical_and(tt_all, flag[:,:,:,3] == 1)
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,0]))
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,1]))
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,2]))
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,3]))
        tt_all = np.logical_and(tt_all, sig[:,:,:,0] != 0)
        tt_all = np.logical_and(tt_all, sig[:,:,:,1] != 0)
        tt_all = np.logical_and(tt_all, sig[:,:,:,2] != 0)
        tt_all = np.logical_and(tt_all, sig[:,:,:,3] != 0)

        vsar1[tt_all,0] =  0.5* (rr[tt_all] + ll[tt_all])   # I = 1/2 (RR + LL)
        vsar1[tt_all,1] =  0.5* (rl[tt_all] + lr[tt_all])   # Q = 1/2 (RL + LR)
        vsar1[tt_all,2] = -0.5j*(rl[tt_all] - lr[tt_all])   # U = 1/2j(RL - LR)
        vsar1[tt_all,3] =  0.5* (rr[tt_all] - ll[tt_all])   # V = 1/2 (RR - LL)

        sig_norm = 0.5*np.sqrt(sig[tt_all,0]**2 + sig[tt_all,1]**2 +
                               sig[tt_all,2]**2 + sig[tt_all,3]**2)
        sig1[tt_all,:] = sig_norm

        flag1[tt_all,:] = 1

        if pseudoI:
            # Find where RL and LR are bad, but
            # either RR or LL are good: tt_ is "truth table"
            # "Finite" means "not Inf and not NaN"

            # Either or both RL or LR are bad
            tt_bad = np.logical_or(flag[:,:,:,2] != 1, flag[:,:,:,3] != 1)

            # Only RR is good
            tt_rr = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] != 1)
            tt_rr = np.logical_and(tt_rr, tt_bad)
            tt_rr = np.logical_and(tt_rr, np.isfinite(sig[:,:,:,0]))
            tt_rr = np.logical_and(tt_rr, sig[:,:,:,0] != 0)
            vsar1[tt_rr,0] = rr[tt_rr]             # I = RR
            #vsar1[tt_rr,1:] = 0.                   # V = 0

            # Only LL is good
            tt_ll = np.logical_and(flag[:,:,:,0] != 1, flag[:,:,:,1] == 1)
            tt_ll = np.logical_and(tt_ll, tt_bad)
            tt_ll = np.logical_and(tt_ll, np.isfinite(sig[:,:,:,1]))
            tt_ll = np.logical_and(tt_ll, sig[:,:,:,1] != 0)
            vsar1[tt_ll,0] = ll[tt_ll]             # I = LL
            #vsar1[tt_ll,1:] = 0.                   # V = 0

            flag1[tt_rr,0] = 1
            flag1[tt_ll,0] = 1


            
    elif lpol == ['XX', 'YY', 'XY', 'YX']:
        xx = vsar[:,:,:,0]
        yy = vsar[:,:,:,1]
        xy = vsar[:,:,:,2]
        yx = vsar[:,:,:,3]

        # Find where XX, YY, XY, and YX are good: tt_ is "truth table"
        # "Finite" means "not Inf and not NaN"

        tt_all = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
        tt_all = np.logical_and(tt_all, flag[:,:,:,2] == 1)
        tt_all = np.logical_and(tt_all, flag[:,:,:,3] == 1)
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,0]))
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,1]))
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,2]))
        tt_all = np.logical_and(tt_all, np.isfinite(sig[:,:,:,3]))
        tt_all = np.logical_and(tt_all, sig[:,:,:,0] != 0)
        tt_all = np.logical_and(tt_all, sig[:,:,:,1] != 0)
        tt_all = np.logical_and(tt_all, sig[:,:,:,2] != 0)
        tt_all = np.logical_and(tt_all, sig[:,:,:,3] != 0)
      
        vsar1[tt_all,0] =  0.5* (xx[tt_all] + yy[tt_all])  # I = 1/2 (XX + YY)
        vsar1[tt_all,1] =  0.5* (xx[tt_all] - yy[tt_all])  # Q = 1/2 (XX - YY)
        vsar1[tt_all,2] =  0.5* (xy[tt_all] + yx[tt_all])  # U = 1/2 (XY + YX)
        vsar1[tt_all,3] = -0.5j*(xy[tt_all] - yx[tt_all])  # V = 1/2j(XY - YX)
        
        sig_norm = 0.5*np.sqrt(sig[tt_all,0]**2 + sig[tt_all,1]**2 +
                               sig[tt_all,2]**2 + sig[tt_all,3]**2)
        sig1[tt_all,:] = sig_norm
        
        flag1[tt_all,:] = 1

        if pseudoI:
            # Find where XY and YX are bad, but
            # either XX or YY are good: tt_ is "truth table"
            # "Finite" means "not Inf and not NaN"

            # Either or both XY or YX are bad
            tt_bad = np.logical_or(flag[:,:,:,2] != 1, flag[:,:,:,3] != 1)

            # Only XX is good
            tt_xx = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] != 1)
            tt_xx = np.logical_and(tt_xx, tt_bad)
            tt_xx = np.logical_and(tt_xx, np.isfinite(sig[:,:,:,0]))
            tt_xx = np.logical_and(tt_xx, sig[:,:,:,0] != 0)
            vsar1[tt_xx,0] = rr[tt_xx]             # I = RR
            #vsar1[tt_xx,1:] = 0.                   # V = 0

            # Only YY is good
            tt_yy = np.logical_and(flag[:,:,:,0] != 1, flag[:,:,:,1] == 1)
            tt_yy = np.logical_and(tt_yy, tt_bad)
            tt_yy = np.logical_and(tt_yy, np.isfinite(sig[:,:,:,1]))
            tt_yy = np.logical_and(tt_yy, sig[:,:,:,1] != 0)
            vsar1[tt_yy,0] = ll[tt_yy]             # I = LL
            #vsar1[tt_yy,1:] = 0.                   # V = 0

            flag1[tt_xx,0] = 1
            flag1[tt_yy,0] = 1









        
    #sys.exit(0)

    #
    # Create visibility table as an xarray.DataArray
    #
    ds1 = Dataset(
        data_vars=dict(
            vis=(["data", "spw", "ch", "pol"], vsar1)
        ),
        coords=dict(
            mjd=("data", vistab.ds.vis.mjd.data),  # .compute()),
            dmjd=("data", vistab.ds.vis.dmjd.data),
            usec=("data", vistab.ds.vis.usec.data),
            vsec=("data", vistab.ds.vis.vsec.data),
            wsec=("data", vistab.ds.vis.wsec.data),
            antid1=("data", vistab.ds.vis.antid1.data),
            antid2=("data", vistab.ds.vis.antid2.data),
            flag=(["data", "spw", "ch", "pol"], flag1),
            sigma=(["data", "spw", "ch", "pol"], sig1),
            pol=(["pol"], ['I', 'Q', 'U', 'V']),
        )
    )

    vt = VisTable(ds=ds1.sortby(["mjd", "antid1", "antid2"]))

    return vt


__switch_polrepr = switch_polrepr





