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
    pseudoI (bool): if True, calculate I from XX or RR.  (??????????)
    
    """
    from .vistab import VisTable



