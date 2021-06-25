#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from .vis.vis_fun import __switch_polrepr


class UVData(object):
    # zarr file
    zarrfile = None

    # Meta data
    ant = None  # antenna meta data
    freq = None  # frequency meta data
    scan = None  # scan information
    src = None  # source information

    # baseline based data
    vis = None  # scan-segmented visibilities
    vistab = None  # uvfits-style baseline-based data

    # bi-spectrum data
    bs = None  # scan-segmented bi-spectra

    # closure-amplitude data
    ca = None  # scan-segmented closure amplitudes

    def __init__(self, zarrfile, ant=None, freq=None, scan=None, src=None,
                 vis=None, vistab=None, bs=None, ca=None):
        self.zarrfile = zarrfile

        if ant is not None:
            self.ant = ant.copy()

        if freq is not None:
            self.freq = freq.copy()

        if scan is not None:
            self.scan = scan.copy()

        if src is not None:
            self.src = src.copy()

        if vis is not None:
            self.vis = vis.copy()

        if vistab is not None:
            self.vistab = vistab.copy()

    def __repr__(self):
        outlines = []
        outlines.append("zarr file: %s" % (self.zarrfile))
        outlines.append("Attributes:")
        if self.ant is not None:
            outlines.append("  ant: antenna-based static metadata")

        if self.freq is not None:
            outlines.append("  freq: frequency setup")

        if self.scan is not None:
            outlines.append("  scan: scan table")

        if self.src is not None:
            outlines.append("  src: source information")

        if self.vis is not None:
            outlines.append("  vis: scan-segmented visibility data set")

        if self.vistab is not None:
            outlines.append("  vistab: uvfits-style visibility data set")
        return "\n".join(outlines)

    @classmethod
    def load_zarr(cls, inzarr):
        from .io.zarr import zarr2UVData
        return zarr2UVData(inzarr)

    def to_zarr(self, outzarr):
        from .io.zarr import UVData2zarr
        UVData2zarr(self, outzarr)

    def calc_vis(self, remove_vistab=True):
        """
        Split vistable into scan-based visibility data sets.

        Args:
            remove_vistab (bool, optional): 
                If True, vistab will be removed after generating vis data set.
                Defaults to True.
        """
        from .vis.vistab2vis import vistab2visds
        vistab2visds(uvd=self, remove_vistab=remove_vistab)



    def switch_polrepr(self, polrepr, pseudoI=False):
        """
        This method changes representation of the polarization of
        the visibility data in the visibility table (of class VisTable).
        It returns a new instance of the VisTable class where the polarization
        is rendered according to the one provided in the polrepr string.

        Args:

        polrepr (str): polarization representation, 
                       "stokes" or "circ" or "linear".
        pseudoI (bool): if True, calculate I from XX or YY or RR or LL.
        """
        vt_new = __switch_polrepr(self.vistab, polrepr, pseudoI)

        return vt_new
        
