#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Module for the Interferometric Array data
'''
__author__ = "Smili Developer Team"


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
# internal
from ..util.table import DataTable, DataSeries, Header


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class Array(object):
    """
    Class for the interferometric array information.

    Attributes:
        name (str): The name of the array.
        table (ArrayTable): The list of antennas in the array.
    """
    name = "SMILI"
    table = None

    def copy(self):
        """
        Replicate the array information to the new Array object.

        Returns:
            Array: copied the array information.
        """
        from copy import deepcopy

        outdata = Array()
        outdata.name = deepcopy(self.name)
        outdata.table = self.table.copy()

        return outdata

    @classmethod
    def load_ehtim_array(cls, arrayobj, name="SMILI", args_load_txt={}):
        """
        Loading an array table in the format of the eht-imaging library.

        Args:
            arrayobj (string or ehtim.array.Array object):
                The input filename or ehtim.array.Array object.
            name (str, default="myarray"):
                The name of the Array.
            args_load_txt (dict, default={}):
                If the filename is specified, this method will use
                ehtim.array.load_txt function. This dictionary is for optinal
                argments of this function.

        Returns:
            Array object: Loaded array information
        """
        import ehtim
        from copy import deepcopy
        from numpy import sqrt, asarray, ones, zeros, nan

        # check the array data
        if isinstance(arrayobj, str):
            array = ehtim.array.load_txt(arrayobj, **args_load_txt)
        elif isinstance(arrayobj, ehtim.array.Array):
            array = deepcopy(arrayobj)
        else:
            raise ValueError("Invalid data type of arrayobj: %s" %
                             (type(arrayobj)))

        tarr = array.tarr.copy()
        Nant = tarr.size
        data = dict(
            antname=asarray(tarr["site"], dtype="U32"),
            x=tarr["x"],
            y=tarr["y"],
            z=tarr["z"],
            sefd1=tarr["sefdr"],
            sefd2=tarr["sefdl"],
            tau1=zeros(Nant),
            tau2=zeros(Nant),
            elmin=ones(Nant),
            elmax=ones(Nant)*90.,
            fr_pa_coeff=tarr["fr_par"],
            fr_el_coeff=tarr["fr_elev"],
            fr_offset=tarr["fr_off"],
            d1=tarr["dr"],
            d2=tarr["dl"],
            anttype=asarray(["ground" for i in range(Nant)], dtype="U8"),
        )

        table = ArrayTable(
            data=data,
            columns=ArrayTable.header.name.to_list()
        )

        idx_tle = sqrt(table["x"]**2+table["y"]**2+table["z"]**2) < 1
        table.loc[idx_tle, ["x", "y", "z"]] = nan
        table.loc[idx_tle, "elmin"] = -90.
        table.loc[idx_tle, "elmax"] = 90.
        table.loc[idx_tle, "anttype"] = "tle"

        arraydata = cls()
        arraydata.name = name
        arraydata.table = table

        return arraydata


class ArrayTable(DataTable):
    """
    This table is storing the antenna summary information.

    Attributes:
        header (Header): the header of this table
    """
    header = Header([
        dict(name="antname", dtype="U32", unit="", comment="Antenna Name"),
        dict(name="x", dtype="float64", unit="m",
             comment="Geocenric Coordinate x"),
        dict(name="y", dtype="float64", unit="m",
             comment="Geocenric Coordinate y"),
        dict(name="z", dtype="float64", unit="m",
             comment="Geocenric Coordinate z"),
        dict(name="sefd1", dtype="float64", unit="Jy",
             comment="Zenith SEFD at Pol 1"),
        dict(name="sefd2", dtype="float64", unit="Jy",
             comment="Zenith SEFD at Pol 2"),
        dict(name="tau1", dtype="float64", unit="",
             comment="Zenith Opacity at Pol 1"),
        dict(name="tau2", dtype="float64", unit="",
             comment="Zenith Opactiy at Pol 2"),
        dict(name="elmin", dtype="float64",
             unit="deg", comment="Minimum Elevation"),
        dict(name="elmax", dtype="float64",
             unit="deg", comment="Maximum Elevation"),
        dict(name="fr_pa_coeff", dtype="float64", unit="",
             comment="Coeffcient for the Parallactic Angle to compute the Field Roation Angle"),
        dict(name="fr_el_coeff", dtype="float64", unit="",
             comment="Coeffcient for the Elevation Angle to compute the Field Roation Angle"),
        dict(name="fr_offset", dtype="float64", unit="deg",
             comment="Offset angle for the Field Roation Angle"),
        dict(name="d1", dtype="float128", unit="", comment="D-term at Pol 1"),
        dict(name="d2", dtype="float128", unit="", comment="D-term at Pol 2"),
        dict(name="anttype", dtype="U8", unit="", comment="Antenna Type"),
    ])

    @property
    def _constructor(self):
        return ArrayTable

    @property
    def _constructor_sliced(self):
        return ArraySeries


class ArraySeries(DataSeries):
    @property
    def _constructor(self):
        return ArraySeries

    @property
    def _constructor_expanddim(self):
        return ArrayTable
