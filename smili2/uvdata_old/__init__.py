#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is the core python module of smili.
The module includes following submodules.
'''
__author__ = "Smili Developper Team"

from .uvdata import UVData
from .uvdata.io.uvfits import uvfits2UVData as load_uvfits
