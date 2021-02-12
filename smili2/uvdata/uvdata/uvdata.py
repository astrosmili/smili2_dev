#!/usr/bin/env python
# -*- coding: utf-8 -*-


class UVData(object):
    # antenna-based data
    ant = None

    # baseline-based data
    vis = None

    # bi-spectrum data
    bs = None

    # closure-amplitude data
    ca = None

    def __init__(self, ant=None, freq=None, src=None, vis=None, bs=None, ca=None):
        if ant is not None:
            self.ant = ant.copy()

        if vis is not None:
            self.vis = vis.copy()

        if freq is not None:
            self.freq = freq.copy()

        if src is not None:
            self.src = src.copy()

        self.bs = bs
        self.ca = ca
