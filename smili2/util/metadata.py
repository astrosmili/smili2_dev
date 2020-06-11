#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This module provides classes to handle scalar-type meta data information for
various classes used in SMILI
'''
__author__ = "Smili Developer Team"

from collections import OrderedDict


class MetaData(OrderedDict):
    """
    A class to handle scalar-type meta data information for various classes.
    This class inherets the OrderedDict class in the Collections module.
    Each element is assumed to be an instance of the MetaRec class.
    """

    def __repr__(self):
        formatstr = "{:<10} {:<20} {:<5} {:<7} {:<}"
        string = formatstr.format("name", "value", "unit", "dtype", "comment")
        for key in self.keys():
            rec = self[key]
            string += "\n"
            string += formatstr.format(key, rec.val,
                                       rec.unit, rec.dtype, rec.comment)
        return string


class MetaRec(object):
    """
    A class to handle scalar-type meta data information.

    Attributes:
        dtype (str): data type interpretable with numpy.dtype
        val (depending of dtype): data value
        unit (str): unit of the value
        comment (str): descriptions about this meta data information
    """
    dtype = None
    val = None
    unit = None
    comment = None

    def __init__(self, val=None, unit=None, dtype=None, comment=None):
        """
        A class to handle scalar-type meta data information.

        Args:
            dtype (str): data type interpretable with numpy.dtype
            val (depending of dtype): data value
            unit (str): unit of the value
            comment (str): descriptions about this meta data information
        """
        self.dtype = dtype
        self.val = val
        self.unit = unit
        self.comment = comment

    def __str__(self):
        from numpy import dtype

        sval = str(self.val)
        unit = self.unit
        comment = self.comment
        #
        # For builtin types str, int and float, use 'str', 'int' and 'float'.
        # For the numpy types, add 'np.' prefix to get type names like
        # 'np.int64' or 'np.float64'.
        #
        dtyp = self.dtype
        if (dtyp is str) or (dtyp is int) or (dtyp is float):
            stype = dtype(self.dtype).name + ', '  # Builtin types
        else:
            stype = 'np.' + dtype(self.dtype).name + ', '
        if unit is not None:
            outstr = stype + ' ' + sval + ' (' + unit + ')'
        else:
            outstr = stype + ' ' + sval
        if comment is not None:
            outstr += ' # ' + comment

        return outstr

    def __repr__(self):
        return self.val.__repr__()
