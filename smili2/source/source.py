#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
__author__ = "Smili Developer Team"


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class Source(object):
    '''A class to handle an astronomical source

    Attributes:
        name (str): the source name
        skycoord (astropy.coord.SkyCoord): the sky coordinate
    '''
    # name
    name = "mysource"
    skycoord = None

    def __init__(self, name, skycoord=None):
        '''Initialize the Source instance
        Args:
            name (str): the source name
            skycoord (astropy.coord.SkyCoord; optional): the source sky
                coordinate. If it is not specified, the coordinate will
                be searched at the CDS.
        '''
        from astropy.coordinates import SkyCoord

        self.name = name

        if skycoord is None:
            self.skycoord = SkyCoord.from_name("M87")
        elif isinstance(skycoord, SkyCoord):
            self.skycoord = skycoord.copy()
        else:
            raise ValueError(
                "skycoord must be an astropy.coord.SkyCoord object")

    def copy(self):
        from copy import deepcopy
        return Source(name=deepcopy(self.name), skycoord=self.skycoord.copy())

    def __repr__(self):
        output = "%s %s" % (self.name, self.skycoord.to_string("hmsdms"))
        return output

    def _repr_html_(self):
        output = "%s %s" % (self.name, self.skycoord.to_string("hmsdms"))
        return output
