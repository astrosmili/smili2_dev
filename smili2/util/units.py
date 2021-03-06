#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This module provides a quick shortcut to the major units.
'''
__author__ = "Smili Developer Team"

from astropy.units import Unit


def conv(from_unit, to_unit):
    """
    Compute the conversion factor between two units.

    Args:
        from_unit (astropy.units.Unit): The unit to be converted from.
        to_unit (astropy.units.Unit): The unit to be converted to.

    Returns:
        float-like: the conversion factor
    """
    factor = (1*Unit(from_unit)) / (1*Unit(to_unit))
    return factor.to_value("")


# Dimension Less
DIMLESS = Unit("")

# Length
M = Unit("m")
MM = Unit("mm")
CM = Unit("cm")
KM = Unit("km")

# Radio Flux
JY = Unit("Jy")
MJY = Unit("mJy")
UJY = Unit("uJy")

# Angular Size
UAS = Unit("uas")
MAS = Unit("mas")
ASEC = Unit("arcsec")
AMIN = Unit("arcmin")
DEG = Unit("deg")
RAD = Unit("rad")
HOURANG = Unit("hourangle")

# Time
SEC = Unit("second")
MIN = Unit("minute")
HOUR = Unit("hour")
DAY = Unit("day")
YEAR = Unit("year")

# Mass
MSUN = Unit("Msun")
