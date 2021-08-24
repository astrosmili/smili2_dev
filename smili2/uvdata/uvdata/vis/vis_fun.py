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
    npol = len(lpol)
    set_pol = set(lpol)

    # Check if the visibolity is already in the requested format
    if polrepr == "stokes" and lpol[0] in ['I', 'Q', 'U', 'V']:
        return vistab # Return the same vistab which is already in Stokes format

    if polrepr == "circ" and lpol[0] in ['RR', 'LL', 'RL', 'LR']:
        return vistab # Return the same vistab which is already in circular

    if polrepr == "linear" and lpol[0] in ['XX', 'YY', 'XY', 'YX']:
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
    sig_norm_01 =  np.zeros(shape1[:-1], dtype=float)
    sig_norm =  np.zeros(shape1[:-1], dtype=float)

    # New polarization coordinates dependent on the requested representation
    if polrepr == "stokes":
        lpol1 = ['I', 'Q', 'U', 'V']
    elif polrepr == "circ":
        lpol1 = ['RR', 'LL', 'RL', 'LR']
    elif polrepr == "linear":
        lpol1 = ['XX', 'YY', 'XY', 'YX']

        
    if polrepr == "stokes" and npol == 1:
        
        # lpol == ['RR'] or lpol == ['LL'] or lpol == ['XX'] or lpol == ['YY']:
        vsar1[:,:,:,0] = np.copy(vsar[:,:,:,0])    # I ~ RR or LL or XX or YY
        sig1[:,:,:,0] =  np.copy(sig[:,:,:,0])
        flag1[:,:,:,0] = np.copy(flag[:,:,:,0])

        
    #
    # Create truth tables
    #

    if polrepr == "stokes" and npol > 1:
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

        # tt_fg01: where first two flags are good (== 1)
        tt_fg01 = np.logical_and(flag[:,:,:,0] == 1, flag[:,:,:,1] == 1)

        # tt_g01: where ALL flags are 1 and ALL sigmas are finite and non-zero
        tt_g01 = np.logical_and(tt_fg01, tt_sg01)

        # tt_fr01: where first two flags are recoverable (== -1)
        tt_fr01 = np.logical_and(flag[:,:,:,0] == -1, flag[:,:,:,1] == -1)
        
        # tt_r01: where both flags are -1 and sigmas are finite and non-zero
        tt_r01 = np.logical_and(tt_fr01, tt_sg01)

        # tt_gr01: where both data are either good or recoverable:
        #         all flags are 1 or -1 and sigmas are finite and non-zero
        tt_gr01 = np.logical_or(tt_g01, tt_r01)

        sig_norm_01[tt_gr01] = 0.5*np.sqrt(sig[tt_gr01,0]**2 + \
                                           sig[tt_gr01,1]**2)


    if polrepr == "stokes" and npol == 4:

        #
        # Create truth table tt_sg01b23:
        #   where first two sigmas are finite and non-zero,
        #   while one or both of the other two are bad
        #
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

        # tt_sg: where all 4 sigmas are finite and non-zero
        tt_sg = np.logical_and(tt_sg01, sig[:,:,:,2] != 0)
        tt_sg = np.logical_and(tt_sg01, sig[:,:,:,3] != 0)


        # tt_fg: where ALL flags are good (== 1)
        tt_fg = np.logical_and(tt_fg01, flag[:,:,:,2] == 1)
        tt_fg = np.logical_and(tt_fg,   flag[:,:,:,3] == 1)

        # tt_g: where ALL flags are 1 and ALL sigmas are finite and non-zero
        tt_g = np.logical_and(tt_fg, tt_sg)


        # tt_fr: where ALL flags are recoverable (== -1)
        tt_fr = np.logical_and(tt_fr01, flag[:,:,:,2] == -1)
        tt_fr = np.logical_and(tt_fr,   flag[:,:,:,3] == -1)

        # tt_r: where ALL flags are -1 and sigmas are finite and non-zero
        tt_r = np.logical_and(tt_fr, tt_sg)

        # tt_gr: where ALL data are either good or recoverable:
        #         all flags are 1 or -1 and sigmas are finite and non-zero
        tt_gr = np.logical_or(tt_g, tt_r)
        
        sig_norm[tt_gr] = 0.5*np.sqrt(sig[tt_gr,0]**2 + sig[tt_gr,1]**2 +
                               sig[tt_gr,2]**2 + sig[tt_gr,3]**2)
        

    if polrepr == "stokes" and pseudoI:

        vsar1[tt_g0b1,0] = vsar[tt_g0b1,0]
        vsar1[tt_b0g1,0] = vsar[tt_b0g1,1]

        flag1[tt_g0b1,0] = 1
        flag1[tt_b0g1,0] = 1

        sig1[tt_g0b1,0] = sig[tt_g0b1,0]  # ??? maybe, sig1[tt_g0b1,0] = ... ?
        sig1[tt_b0g1,0] = sig[tt_b0g1,1]  # ??? maybe, sig1[tt_b0g1,0] = ... ?
        
    #
    # Compute Stokes' visibilities for different polarizations
    #
        
    if polrepr == "stokes" and set_pol == set(['RR', 'LL']):
        
        rr = vsar[:,:,:,0]
        ll = vsar[:,:,:,1]

        vsar1[tt_gr01,0] = 0.5*(rr[tt_gr01] + ll[tt_gr01]) # I = 1/2 (RR + LL)
        vsar1[tt_gr01,3] = 0.5*(rr[tt_gr01] - ll[tt_gr01]) # V = 1/2 (RR - LL)

        flag1[tt_g01,0] = 1
        flag1[tt_g01,3] = 1

        flag1[tt_r01,0] = -1
        flag1[tt_r01,3] = -1

        sig1[tt_gr01,0] = sig_norm_01[tt_gr01]
        sig1[tt_gr01,3] = sig_norm_01[tt_gr01]

           
    elif polrepr == "stokes" and set_pol == set(['XX', 'YY']):
        
        xx = vsar[:,:,:,0]
        yy = vsar[:,:,:,1]

        vsar1[tt_gr01,0] = 0.5*(xx[tt_gr01] + yy[tt_gr01])  # I = 1/2 (XX + YY)
        vsar1[tt_gr01,1] = 0.5*(xx[tt_gr01] - yy[tt_gr01])  # Q = 1/2 (XX - YY)

        flag1[tt_g01,0] = 1
        flag1[tt_g01,1] = 1

        flag1[tt_r01,0] = -1
        flag1[tt_r01,1] = -1

        sig1[tt_gr01,0] = sig_norm_01[tt_gr01]
        sig1[tt_gr01,1] = sig_norm_01[tt_gr01]
    
            
    elif polrepr == "stokes" and set_pol == set(['RR', 'LL', 'RL', 'LR']):
        
        rr = vsar[:,:,:,0]
        ll = vsar[:,:,:,1]
        rl = vsar[:,:,:,2]
        lr = vsar[:,:,:,3]

        vsar1[tt_g01b23,0] =  0.5*(rr[tt_g01b23] + ll[tt_g01b23])  # I
        vsar1[tt_g01b23,3] =  0.5*(rr[tt_g01b23] - ll[tt_g01b23])  # V

        flag1[tt_g01b23,0] = 1         # I
        flag1[tt_g01b23,3] = 1         # V

        sig1[tt_g01b23,0] = sig_norm_01[tt_g01b23]
        sig1[tt_g01b23,3] = sig_norm_01[tt_g01b23]

        vsar1[tt_gr,0] =  0.5* (rr[tt_gr] + ll[tt_gr])   # I = 1/2 (RR + LL)
        vsar1[tt_gr,1] =  0.5* (rl[tt_gr] + lr[tt_gr])   # Q = 1/2 (RL + LR)
        vsar1[tt_gr,2] = -0.5j*(rl[tt_gr] - lr[tt_gr])   # U = 1/2j(RL - LR)
        vsar1[tt_gr,3] =  0.5* (rr[tt_gr] - ll[tt_gr])   # V = 1/2 (RR - LL)

        flag1[tt_g,:] = 1
        flag1[tt_r,:] = -1

        for ipol in range(4): sig1[tt_gr,ipol] = sig_norm[tt_gr]
        

    elif polrepr == "stokes" and set_pol == set(['XX', 'YY', 'XY', 'YX']):
        
        xx = vsar[:,:,:,0]
        yy = vsar[:,:,:,1]
        xy = vsar[:,:,:,2]
        yx = vsar[:,:,:,3]

        vsar1[tt_g01b23,0] =  0.5*(xx[tt_g01b23] + yy[tt_g01b23])  # I
        vsar1[tt_g01b23,1] =  0.5*(xx[tt_g01b23] - yy[tt_g01b23])  # Q

        flag1[tt_g01b23,0] = 1         # I
        flag1[tt_g01b23,1] = 1         # Q

        sig1[tt_g01b23,0] = sig_norm_01[tt_g01b23]
        sig1[tt_g01b23,1] = sig_norm_01[tt_g01b23]

        vsar1[tt_gr,0] =  0.5* (xx[tt_gr] + yy[tt_gr])  # I = 1/2 (XX + YY)
        vsar1[tt_gr,1] =  0.5* (xx[tt_gr] - yy[tt_gr])  # Q = 1/2 (XX - YY)
        vsar1[tt_gr,2] =  0.5* (xy[tt_gr] + yx[tt_gr])  # U = 1/2 (XY + YX)
        vsar1[tt_gr,3] = -0.5j*(xy[tt_gr] - yx[tt_gr])  # V = 1/2j(XY - YX)
        
        flag1[tt_g,:] = 1
        flag1[tt_r,:] = -1

        for ipol in range(4): sig1[tt_gr,ipol] = sig_norm[tt_gr]
        

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


def average_time(vistab, new_tint):
    
    
    #
    # Extract visibility, flag, and sigma arrays
    #
    vsar = vistab.ds.vis.data       #.compute()   # Extract visibility array
    flag = vistab.ds.vis.flag.data  #.compute() 
    sig =  vistab.ds.vis.sigma.data.compute()     # Extract sigma array

    shape1 = list(vsar.shape)  # Dimensions [data, spw, ch, pol]
    
    # # New visibility ndarrays for the visibility, flags and sigmas
    # vsar1 = np.zeros(shape1, dtype=complex)
    # flag1 = np.zeros(shape1, dtype=np.int32) # Assume ALL the data are bad
    # sig1 =  np.zeros(shape1, dtype=float)
    # sig_norm_01 =  np.zeros(shape1[:-1], dtype=float)
    # sig_norm =  np.zeros(shape1[:-1], dtype=float)





