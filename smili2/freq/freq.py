#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
__author__ = "Smili Developer Team"


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
# internal
from ..util.table import DataTable, DataSeries, TableHeader


# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
class Freq(object):
    name = "SMILI"
    table = None
    Nch = 1

    def __init__(self, freqtable, Nch=1, name="SMILI"):
        if isinstance(freqtable, FreqTable):
            self.table = freqtable
        else:
            self.table = FreqTable(freqtable)[FreqTable.header.name.to_list()]

        self.name = name
        self.Nch = Nch

    def __repr__(self):
        # get the number of if and ch
        Nif, Nch = self.get_shape()

        output = ""
        output += "Frequency Setup: %s (Nif: %d, Nch: %d)\n" % (
            self.name, Nif, Nch)
        output += self.table.__repr__()

        return output

    def get_freqarr(self):
        from numpy import zeros, arange

        # reset index
        self.table.reset_index(drop=True, inplace=True)
        table = self.table

        # get the number of if and ch
        Nif, Nch = self.get_shape()

        # create an array
        freqarr = zeros([Nif, Nch], dtype="float64")
        chidarr = arange(Nch)

        # compute frequency
        for iif in range(Nif):
            if_freq = table.loc[iif, "if_freq"]
            sideband = table.loc[iif, "sideband"]
            ch_bw = table.loc[iif, "ch_bw"]
            freqarr[iif] = if_freq + sideband * ch_bw * chidarr

        return freqarr

    def get_reffreq(self, mode="min"):
        if "min" in mode.lower():
            return self.get_freqarr().min()
        elif "mean" in mode.lower():
            return self.get_freqarr().mean()
        elif "median" in mode.lower():
            from numpy import median
            return median(self.get_freqarr())
        elif "max" in mode.lower():
            return self.get_freqarr().max()
        else:
            raise ValueError("mode=%s is not an available option." % (mode))

    def get_shape(self):
        """
        Get the shape of the frequency structure.

        Returns:
            tuple: (Nif, Nch)
        """
        return (len(self.table), self.Nch)

    def copy(self):
        """
        Replicate the array information to the new Array object.

        Returns:
            Array: copied the array information.
        """
        from copy import deepcopy

        outdata = Freq(
            freqtable=self.table.copy(),
            name=deepcopy(self.name),
            Nch=deepcopy(self.Nch)
        )

        return outdata

    def to_text(self, filename=None):
        """
        Output the frequency information to an ascii file in the SMILI format.

        Args:
            filename (str): output filename
        """
        Nch = self.Nch

        header = "# SMILI Frequency Table\n"
        header += "#\n"
        header += "#   Metadata:\n"
        header += "#     name: %s\n" % (self.name)
        header += "#     n_ch: %d\n" % (Nch)
        header += "#\n"
        header += "#   Columns:\n"
        header += "#     %8s %8s %4s %s\n" % ("name",
                                              "dtype", "unit", "comment")
        for irow in range(len(self.table.header)):
            c1, c2, c3, c4 = self.table.header.loc[irow]
            header += "#     %8s %8s %4s %s\n" % (c1, c2, c3, c4)
        header += "#\n"

        if filename is not None:
            f = open(filename, "w")
            f.write(header)
            self.table.to_string(f, float_format="%g", index=False)
            f.close()
        else:
            return filename

    @classmethod
    def load_text(cls, filename):
        """
        Load the frequency information to an ascii file in the SMILI format.

        Args:
            filename (str): input filename
        """
        import re
        from io import StringIO
        import pandas as pd

        # patern for commented out lines
        p = re.compile(r"^\s*#")

        # read text files
        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        # get data lines and other metadata
        datalines = []
        name = None
        n_ch = None
        for line in lines:
            if not p.findall(line):
                datalines.append(line)
            else:
                if "name:" in line:
                    line_tmp = p.split(line)[1]
                    line_tmp = line_tmp.replace("\n", "")
                    line_tmp = line_tmp.replace(":", " ")
                    while "  " in line_tmp:
                        line_tmp = line_tmp.replace("  ", " ")
                    line_tmp = line_tmp.replace(" name", "name")
                    name = line_tmp.split(" ")[1:]
                    name = " ".join(name)
                elif "n_ch:" in line:
                    line_tmp = p.split(line)[1]
                    line_tmp = line_tmp.replace("\n", "")
                    line_tmp = line_tmp.replace(":", " ")
                    while "  " in line_tmp:
                        line_tmp = line_tmp.replace("  ", " ")
                    line_tmp = line_tmp.replace(" n_ch", "n_ch")
                    n_ch = int(line_tmp.split(" ")[1])

        if name is None:
            name = "SMILI"

        if n_ch is None:
            raise ValueError(
                "The input file does not include the number of channels in its header.")

        s = StringIO("".join(datalines))
        freqtable = FreqTable(pd.read_csv(s, sep=r"\s+", index_col=False))
        s.close()

        return cls(freqtable=freqtable, Nch=n_ch, name=name)


class FreqTable(DataTable):
    header = TableHeader([
        dict(name="if_freq", dtype="float64", unit="Hz",
             comment="Central frequency of the 1st channel of the IF"),
        dict(name="ch_bw", dtype="float64", unit="Hz",
             comment="Channel Bandwidth"),
        dict(name="sideband", dtype="int32", unit="",
             comment="Sideband (1=USB, -1=LSB)"),
    ])

    @property
    def _constructor(self):
        return FreqTable

    @property
    def _constructor_sliced(self):
        return FreqSeries


class FreqSeries(DataSeries):
    @property
    def _constructor(self):
        return FreqSeries

    @property
    def _constructor_expanddim(self):
        return FreqTable
