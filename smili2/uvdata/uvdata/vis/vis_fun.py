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

    #
    # Sanity check
    #
    if polrepr != "stokes" and polrepr != "circ" and polrepr != "linear":
        print('ERROR. switch_polrepr(): parameter polrepr must be "stokes" ' \
              'or "circ" or "linear"')
        return None
    

    lpol = list(vistab.ds.vis.pol.data) # List of pols like ['RR', 'LL']
    lpol = [pol.upper() for pol in lpol]
    set_pol = set(lpol)

    # Check if the visibolity is already in the requested format
    if polrepr == "stokes" and lpol[0] in ['I', 'Q', 'U', 'V']:
        return vistab # Return the same vistab which is already in Stokes format

    if polrepr == "circ" and lpol[0] in ['RR', 'LL', 'RL', 'LR']:
        return vistab # Return the same vistab which is already in circular

    if polrepr == "linear" and lpol[0] == ['XX', 'YY', 'XY', 'YX']:
        return vistab # Return the same vistab which is already in linear

    #
    # Extract visibility, flag, and sigma arrays
    #
    vsar = vistab.ds.vis.data       #.compute()   # Extract visibility array
    flag = vistab.ds.vis.flag.data  #.compute() 
    sig =  vistab.ds.vis.sigma.data.compute()     # Extract sigma array

    shape1 = list(vsar.shape)  # Dimensions [data, spw, ch, pol]
    shape1[-1] = 4          # Extend pol axis to 4 to hold 4 Stokes parameters
    
    # New visibility ndarrays for the visibility, flags and sigmas
    vsar1 = np.zeros(shape1, dtype=complex)
    flag1 = np.zeros(shape1, dtype=np.int32) # Assume ALL the data are bad
    sig1 =  np.zeros(shape1, dtype=float)

    # New polarization coordinates dependent on the requested representation
    if polrepr == "stokes":
        lpol1 = ['I', 'Q', 'U', 'V']
    elif polrepr == "circ":
        lpol1 = ['RR', 'LL', 'RL', 'LR']
    elif polrepr == "linear":
        lpol1 = ['XX', 'YY', 'XY', 'YX']

    if polrepr == "stokes" and \
       (lpol == ['RR'] or lpol == ['LL'] or lpol == ['XX'] or lpol == ['YY']):
        
        vsar1[:,:,:,0] = np.copy(vsar[:,:,:,0])    # I ~ RR or LL or XX or YY
        #vsar1[:,:,:,1:] = 0.
        sig1[:,:,:,0] = np.copy(sig[:,:,:,0])
        #sig1[:,:,:,1:] = 0.
        flag1[:,:,:,0] = np.copy(flag[:,:,:,0])
        #flag1[:,:,:,1:] = 0.


    elif polrepr == "stokes" and set_pol == set(['RR', 'LL']):
        
        rr = vsar[:,:,:,0]
        ll = vsar[:,:,:,1]

        # tt_fnz: where sigmas are finite and non-zero
        tt_fnz = np.logical_and(np.isfinite(sig[:,:,:,0]),
                                np.isfinite(sig[:,:,:,1]))
        tt_fnz = np.logical_and(tt_fnz, sig[:,:,:,0] != 0)
        tt_fnz = np.logical_and(tt_fnz, sig[:,:,:,1] != 0)

        # tt_all1: where both RR and LL are good:
        #          both flags are 1 and sigmas are finite and non-zero
        tt_all1 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
        tt_all1 = np.logical_and(tt_all1, tt_fnz)

        # tt_all2: where both RR and LL are recoverable:
        #          both flags are -1 and sigmas are finite and non-zero
        tt_all2 = np.logical_and(flag[:,:,:,0] == -1, flag[:,:,:,1] == -1)
        tt_all2 = np.logical_and(tt_all2, tt_fnz)

        # tt_all: where both RR and LL are either good or recoverable:
        #         both flags are 1 or -1 and sigmas are finite and non-zero
        tt_all = np.logical_or(tt_all1, tt_all2)

        flag1[tt_all1,0] = 1
        flag1[tt_all1,3] = 1

        flag1[tt_all2,0] = -1
        flag1[tt_all2,3] = -1

        vsar1[tt_all,0] = 0.5*(rr[tt_all] + ll[tt_all])    # I = 1/2 (RR + LL)
        #vsar1[tt_all,1] = 0.                              # Q = 1/2 (RL + LR)
        #vsar1[tt_all,2] = 0.                              # U = 1/2j(RL - LR)
        vsar1[tt_all,3] = 0.5*(rr[tt_all] - ll[tt_all])    # V = 1/2 (RR - LL)

        sig_norm = 0.5*np.sqrt(sig[tt_all,0]**2 + sig[tt_all,1]**2)
        sig1[tt_all,0] = sig_norm
        sig1[tt_all,3] = sig_norm

        if pseudoI:
            # Find where either RR or LL are good: tt_ is "truth table"
            # "Finite" means "not Inf and not NaN"

            tt_rr = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] != 1)
            tt_rr = np.logical_and(tt_rr, np.isfinite(sig[:,:,:,0]))
            tt_rr = np.logical_and(tt_rr, sig[:,:,:,0] != 0)
            vsar1[tt_rr,0] = rr[tt_rr]             # I = RR
            #vsar1[tt_rr,3] = 0.                   # V = 0

            tt_ll = np.logical_and(flag[:,:,:,0] != 1, flag[:,:,:,1] == 1)
            tt_ll = np.logical_and(tt_ll, np.isfinite(sig[:,:,:,1]))
            tt_ll = np.logical_and(tt_ll, sig[:,:,:,1] != 0)
            vsar1[tt_ll,0] = ll[tt_ll]             # I = LL
            #vsar1[tt_ll,3] = 0.                   # V = 0

            flag1[tt_rr,0] = 1
            flag1[tt_ll,0] = 1

            
    elif polrepr == "stokes" and set_pol == set(['XX', 'YY']):
        
        xx = vsar[:,:,:,0]
        yy = vsar[:,:,:,1]

        # tt_fnz: where sigmas are finite and non-zero
        tt_fnz = np.logical_and(np.isfinite(sig[:,:,:,0]),
                                np.isfinite(sig[:,:,:,1]))
        tt_fnz = np.logical_and(tt_fnz, sig[:,:,:,0] != 0)
        tt_fnz = np.logical_and(tt_fnz, sig[:,:,:,1] != 0)

        # tt_all1: where both XX and YY are good:
        #          both flags are 1 and sigmas are finite and non-zero
        tt_all1 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
        tt_all1 = np.logical_and(tt_all1, tt_fnz)

        # tt_all2: where both XX and YY are recoverable:
        #          both flags are -1 and sigmas are finite and non-zero
        tt_all2 = np.logical_and(flag[:,:,:,0] == -1, flag[:,:,:,1] == -1)
        tt_all2 = np.logical_and(tt_all2, tt_fnz)

        # tt_all: where both XX and YY are either good or recoverable:
        #         both flags are 1 or -1 and sigmas are finite and non-zero
        tt_all = np.logical_or(tt_all1, tt_all2)

        flag1[tt_all1,0] = 1
        flag1[tt_all1,1] = 1

        flag1[tt_all2,0] = -1
        flag1[tt_all2,1] = -1

        vsar1[tt_all,0] = 0.5*(xx[tt_all] + yy[tt_all])  # I = 1/2 (XX + YY)
        vsar1[tt_all,1] = 0.5*(xx[tt_all] - yy[tt_all])  # Q = 1/2 (XY + YX)
        #vsar1[tt_all,2] = 0.                            # U = 1/2j(XY - YX)
        #vsar1[tt_all,3] = 0.                            # V = 1/2 (XX - YY)

        sig_norm = 0.5*np.sqrt(sig[tt_all,0]**2 + sig[tt_all,1]**2)
        sig1[tt_all,0] = sig_norm
        sig1[tt_all,3] = sig_norm
    
        if pseudoI:
            # Find where either XX or YY are good: tt_ is "truth table"
            # "Finite" means "not Inf and not NaN"

            tt_xx = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] != 1)
            tt_xx = np.logical_and(tt_xx, np.isfinite(sig[:,:,:,0]))
            tt_xx = np.logical_and(tt_xx, sig[:,:,:,0] != 0)
            vsar1[tt_xx,0] = xx[tt_xx]             # I = XX
            #vsar1[tt_xx,3] = 0.                   # V = 0

            tt_yy = np.logical_and(flag[:,:,:,0] != 1, flag[:,:,:,1] == 1)
            tt_yy = np.logical_and(tt_yy, np.isfinite(sig[:,:,:,1]))
            tt_yy = np.logical_and(tt_yy, sig[:,:,:,1] != 0)
            vsar1[tt_yy,0] = yy[tt_yy]             # I = YY
            #vsar1[tt_yy,3] = 0.                   # V = 0

            flag1[tt_xx,0] = 1
            flag1[tt_yy,0] = 1

            
    elif polrepr == "stokes" and set_pol == set(['RR', 'LL', 'RL', 'LR']):
        
        rr = vsar[:,:,:,0]
        ll = vsar[:,:,:,1]
        rl = vsar[:,:,:,2]
        lr = vsar[:,:,:,3]

        #
        # Create truth table tt_sg01b23:
        #   where first two sigmas are finite and non-zero,
        #   while one or both of the other two are bad
        #
        
        # tt_sg01: where first two sigmas are good (finite and non-zero)
        tt_sg01 = np.logical_and(np.isfinite(sig[:,:,:,0]),
                                  np.isfinite(sig[:,:,:,1]))
        tt_sg01 = np.logical_and(tt_sg01, sig[:,:,:,0] != 0)
        tt_sg01 = np.logical_and(tt_sg01, sig[:,:,:,1] != 0) # 0th & 1st good
        
        # tt_sg23: where two last sigmas are good (finite and non-zero)
        tt_sg23 = np.logical_and(np.isfinite(sig[:,:,:,2]),
                                  np.isfinite(sig[:,:,:,3]))
        tt_sg23 = np.logical_and(tt_sg23, sig[:,:,:,2] != 0)
        tt_sg23 = np.logical_and(tt_sg23, sig[:,:,:,3] != 0) # 2nd & 3rd good

        # tt_sb23: where one or both 2nd & 3rd are bad
        tt_sb23 = np.logical_not(tt_sg23) 
        
        # tt_sg01b23: where first two sigmas are good (finite and non-zero),
        #               while one or both of the other two are bad
        tt_sg01b23 = np.logical_and(tt_sg01, tt_sb23)

        
        #
        # Create truth table tt_fg01b23:
        #   where first two flags are good (== 1),
        #   while one or both of the other two are bad (!= 1)
        #
        
        # tt_fg01: where first two flags are good (== 1)
        tt_fg01 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
               
        # tt_fb23: one or both flags 2, 3 are bad (!= 1)
        tt_fb23 = np.logical_or(flag[:,:,:,2] != 1, flag[:,:,:,3] != 1)
        
        # tt_fg01b23: where first two flags are good (== 1),
        #             while one or both of the other two are bad (!= 1)
        tt_fg01b23 = np.logical_and(tt_fg01, tt_fb23)

        
        #
        # Create truth table tt_g01b23:
        #   where first two flags and sigmas are good,
        #   while one or both of the other two flags and sigmas are bad
        #
        tt_g01b23 = np.logical_and(tt_sg01b23, tt_fg01b23)

        flag1[tt_g01b23,0] = 1         # I
        # flag1[tt_g01b23,1] = 0         # Q
        # flag1[tt_g01b23,2] = 0         # U
        flag1[tt_g01b23,3] = 1         # V

        vsar1[tt_g01b23,0] =  0.5* (rr[tt_g01b23] + ll[tt_g01b23])  # I
        vsar1[tt_g01b23,1] =  0                                     # Q
        vsar1[tt_g01b23,2] =  0                                     # U
        vsar1[tt_g01b23,3] =  0.5* (rr[tt_g01b23] - ll[tt_g01b23])  # V

        sig_norm = 0.5*np.sqrt(sig[tt_g01b23,0]**2 + sig[tt_g01b23,3]**2)
        sig1[tt_g01b23,:] = sig_norm

#-------------------------------------------------------------------

        
        # tt_sg: where ALL sigmas are finite and non-zero
        tt_sg = np.logical_and(       np.isfinite(sig[:,:,:,0]),
                                      np.isfinite(sig[:,:,:,1]))
        tt_sg = np.logical_and(tt_sg, np.isfinite(sig[:,:,:,2]))
        tt_sg = np.logical_and(tt_sg, np.isfinite(sig[:,:,:,3]))
        
        tt_sg = np.logical_and(tt_sg, sig[:,:,:,0] != 0)
        tt_sg = np.logical_and(tt_sg, sig[:,:,:,1] != 0)
        tt_sg = np.logical_and(tt_sg, sig[:,:,:,2] != 0)
        tt_sg = np.logical_and(tt_sg, sig[:,:,:,3] != 0)


        # tt_fg: where ALL flags are good (== 1)
        tt_fg = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
        tt_fg = np.logical_and(tt_fg,              flag[:,:,:,2] == 1)
        tt_fg = np.logical_and(tt_fg,              flag[:,:,:,3] == 1)

        # tt_g: where ALL flags are 1 and ALL sigmas are finite and non-zero
        tt_g = np.logical_and(tt_fg, tt_sg)

        
        # tt_fr: where ALL flags are recoverable (== -1)
        tt_fr = np.logical_and(flag[:,:,:,0] == -1, flag[:,:,:,1] == -1)
        tt_fr = np.logical_and(tt_fr,               flag[:,:,:,2] == -1)
        tt_fr = np.logical_and(tt_fr,               flag[:,:,:,3] == -1)

        # tt_r: where ALL flags are -1 and sigmas are finite and non-zero
        tt_r = np.logical_and(tt_fr, tt_sg)
        
        # tt_gr: where ALL data are either good or recoverable:
        #         all flags are 1 or -1 and sigmas are finite and non-zero
        tt_gr = np.logical_or(tt_g, tt_r)


        flag1[tt_g,:] = 1
        flag1[tt_r,:] = -1

        vsar1[tt_gr,0] =  0.5* (rr[tt_gr] + ll[tt_gr])   # I = 1/2 (RR + LL)
        vsar1[tt_gr,1] =  0.5* (rl[tt_gr] + lr[tt_gr])   # Q = 1/2 (RL + LR)
        vsar1[tt_gr,2] = -0.5j*(rl[tt_gr] - lr[tt_gr])   # U = 1/2j(RL - LR)
        vsar1[tt_gr,3] =  0.5* (rr[tt_gr] - ll[tt_gr])   # V = 1/2 (RR - LL)

        sig_norm = 0.5*np.sqrt(sig[tt_gr,0]**2 + sig[tt_gr,1]**2 +
                               sig[tt_gr,2]**2 + sig[tt_gr,3]**2)
        sig1[tt_gr,:] = sig_norm

        if pseudoI:
            # tt_g0b1: where pol 0 is good and pol 1 is bad
            tt_fg0b1 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] != 1)
            tt_sg0 = np.logical_and(np.isfinite(sig[:,:,:,0]),
                                                sig[:,:,:,0] != 0)
            tt_sg1 = np.logical_and(np.isfinite(sig[:,:,:,1]),
                                                sig[:,:,:,1] != 0)
            tt_sb1 = np.logical_not(tt_sg1)
            tt_sg0b1 = np.logical_and(tt_sg0, tt_sb1)
            tt_g0b1 = np.logical_and(tt_fg0b1, tt_sg0b1)
            
            vsar1[tt_g0b1,0] = rr[tt_g0b1]             # I = RR
            #vsar1[tt_g0b1,1:] = 0.                  # V = 0

            # tt_b0g1: where pol 0 is good and pol 1 is bad
            tt_fb0g1 = np.logical_and(flag[:,:,:,0] != 1, flag[:,:,:,1] == 1)
            tt_sg1 = np.logical_and(np.isfinite(sig[:,:,:,1]),
                                                sig[:,:,:,1] != 0)
            tt_g1 = np.logical_and(tt_fb0g1, tt_sg1)
            
            vsar1[tt_g1,0] = ll[tt_g1]             # I = RR
            #vsar1[tt_g0,1:] = 0.                  # V = 0

            flag1[tt_g0,0] = 1
            flag1[tt_g1,0] = 1

            
    elif polrepr == "stokes" and set_pol == set(['XX', 'YY', 'XY', 'YX']):
        
        xx = vsar[:,:,:,0]
        yy = vsar[:,:,:,1]
        xy = vsar[:,:,:,2]
        yx = vsar[:,:,:,3]

        # tt_fnz: where all sigmas are finite and non-zero
        tt_fnz = np.logical_and(np.isfinite(sig[:,:,:,0]),
                                np.isfinite(sig[:,:,:,1]))
        tt_fnz = np.logical_and(tt_fnz, np.isfinite(sig[:,:,:,2]))
        tt_fnz = np.logical_and(tt_fnz, np.isfinite(sig[:,:,:,3]))
        
        tt_fnz = np.logical_and(tt_fnz, sig[:,:,:,0] != 0)
        tt_fnz = np.logical_and(tt_fnz, sig[:,:,:,1] != 0)
        tt_fnz = np.logical_and(tt_fnz, sig[:,:,:,2] != 0)
        tt_fnz = np.logical_and(tt_fnz, sig[:,:,:,3] != 0)

        # tt_gr1: where XX, YY, XY, and YX are good and
        #          all flags are 1 and sigmas are finite and non-zero
        tt_all1 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)
        tt_all1 = np.logical_and(tt_all1, flag[:,:,:,2] == 1)
        tt_all1 = np.logical_and(tt_all1, flag[:,:,:,3] == 1)
        tt_all1 = np.logical_and(tt_all1, tt_fnz)

        # tt_all2: where XX, YY, XY, and YX are recoverable and
        #          all flags are -1 and sigmas are finite and non-zero
        tt_all2 = np.logical_and(flag[:,:,:,0] == -1, flag[:,:,:,1] == -1)
        tt_all2 = np.logical_and(tt_all2, flag[:,:,:,2] == -1)
        tt_all2 = np.logical_and(tt_all2, flag[:,:,:,3] == -1)
        tt_all2 = np.logical_and(tt_all2, tt_fnz)

        # tt_all: where XX, YY, XY, and YX are either good or recoverable:
        #         both flags are 1 or -1 and sigmas are finite and non-zero
        tt_all = np.logical_or(tt_all1, tt_all2)

        flag1[tt_all1,:] = 1
        flag1[tt_all2,:] = -1

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
            #vsar1[tt_xx,1:] = 0.                  # V = 0

            # Only YY is good
            tt_yy = np.logical_and(flag[:,:,:,0] != 1, flag[:,:,:,1] == 1)
            tt_yy = np.logical_and(tt_yy, tt_bad)
            tt_yy = np.logical_and(tt_yy, np.isfinite(sig[:,:,:,1]))
            tt_yy = np.logical_and(tt_yy, sig[:,:,:,1] != 0)
            vsar1[tt_yy,0] = ll[tt_yy]             # I = LL
            #vsar1[tt_yy,1:] = 0.                  # V = 0

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
            pol=(["pol"], lpol1),
        )
    )

    vt = VisTable(ds=ds1.sortby(["mjd", "antid1", "antid2"]))

    return vt


__switch_polrepr = switch_polrepr





