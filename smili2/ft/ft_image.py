#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule to handle Fourier Tranformation between images and visibilities.
'''
__author__ = "Smili Developer Team"


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
# numerical packages
import numpy as np

# Logger
from logging import getLogger
logger = getLogger(__name__)

# ------------------------------------------------------------------------------
# Class for the Fourier Transformation
# ------------------------------------------------------------------------------


class NFFT_Image(object):
    '''
    A class of NFFT functions for 2D image
    '''

    def __init__(self, u, v, dx, nx, dy=None, ny=None, ftsign=1):
        '''
        Initialize the Fourier Transform functions

        Args:
            u, v (1d numpy array):
                spatial frequencies in lambda
            dx, dy (float):
                The pixel sizes of input images along RA, Dec axises,
                respectively, in radians
            nx, ny (integer):
                The number of pixels of input images along RA, Dec axises,
                respectively.
            ftsign (integer; default=1):
                The sign of the Fourier exponent. It seems that the positive is
                the standard for the radio astronomy convension.
            kernelfunc (function; default=None):
                If specified, the kernel function will be multipled or devided
                after/before forward/inverse FFT, respectively. This should
                have a form of kernelfunc(u,v), returning full complex
                visibilities.
        Returns:
            FT_Image object
        '''
        from pynfft.nfft import NFFT

        # Sanity Check
        #   size of uv coordinates
        if u.size != v.size:
            raise ValueError("u and v must have the same size")
        #   y values
        if dy is None:
            dy = np.abs(dx)
        if ny is None:
            ny = nx
        #   sign of dx or dy
        if dx > 0:
            logger.warning(
                "The pixel increment dx for RA is positive. (usually negative for astronomical images)")
        if dy < 0:
            logger.warning(
                "The pixel increment dy for Dec is negative. (usually positive for astronomical images)")
        #   sign of the Fourier exponent
        if np.sign(ftsign) <= 0:
            logger.warning("Non standard sign of the Fourier exponent.")

        # the size of u & v vectors
        Nuv = u.size

        # scale u, v coordinates
        u_nfft = u * dx
        v_nfft = v * dy
        # the negative factor is coming from the fact that the NFFT's Fourier
        # exponent sign is opposite to the radio astronomy convension.
        # see NFFT's documentation.
        if np.sign(ftsign) > 0:
            u_nfft *= -1
            v_nfft *= -1

        # initialize nfft routines
        self.plan = NFFT([ny, nx], Nuv, d=2)
        self.plan.x = np.vstack([v_nfft, u_nfft]).T
        self.plan.precompute()

        # set forward / inverse FT functions
        self.forward = self.nfft2d_forward
        self.adjoint = self.nfft2d_adjoint
        self.adjoint_real = self.nfft2d_adjoint_real

    def nfft2d_forward(self, I2d):
        '''
        Two-dimensional Forward Non-uniform Fast Fourier Transform

        Args:
            image in two dimensional numpy array

        Returns:
            complex visibilities in one dimensional numpy array
        '''
        self.plan.f_hat = I2d
        return self.plan.trafo()

    def nfft2d_adjoint(self, Vcmp):
        '''
        Two-dimensional Adjoint Non-uniform Fast Fourier Transform

        Args:
            complex visibilities in one dimensional numpy array

        Returns:
            image in two dimensional numpy array
        '''
        self.plan.f = Vcmp
        return self.plan.adjoint()

    def nfft2d_adjoint_real(self, Vcmp):
        '''
        Two-dimensional Adjoint Non-uniform Fast Fourier Transform

        Args:
            complex visibilities in one dimensional numpy array

        Returns:
            image in two dimensional numpy array
        '''
        self.plan.f = Vcmp
        return np.real(self.plan.adjoint())
