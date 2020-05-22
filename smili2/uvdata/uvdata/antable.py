#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
__author__ = "Smili Developer Team"


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
# internal
from ...util.table import DataTable, DataSeries, Header


# ------------------------------------------------------------------------------
# functions
# ------------------------------------------------------------------------------
class ANTable(DataTable):
    header = Header([
        dict(name="antname", dtype="U32", unit="", comment="Antenna Name"),
        dict(name="antid", dtype="int32", unit="", comment="Antenna ID"),
        dict(name="mjd", dtype="float64", unit="day",
             comment="Modified Jurian Day"),
        dict(name="gst", dtype="float64", unit="hourangle",
             comment="Greenwich Sidereal Time"),
        dict(name="ra", dtype="float64", unit="deg",
             comment="GCRS Right Ascention"),
        dict(name="dec", dtype="float64", unit="deg", comment="GCRS Declination"),
        dict(name="x", dtype="float64", unit="m",
             comment="Geocenric Coordinate x"),
        dict(name="y", dtype="float64", unit="m",
             comment="Geocenric Coordinate y"),
        dict(name="z", dtype="float64", unit="m",
             comment="Geocenric Coordinate z"),
        dict(name="az", dtype="float64", unit="deg", comment="Azimuthal Angle"),
        dict(name="el", dtype="float64", unit="deg", comment="Elevation Angle"),
        dict(name="par", dtype="float64", unit="deg",
             comment="Parallactic Angle"),
        dict(name="fra", dtype="float64", unit="deg",
             comment="Field Roation Angle"),
        dict(name="sefd1", dtype="float64", unit="Jy", comment="SEFD at Pol 1"),
        dict(name="sefd2", dtype="float64", unit="Jy", comment="SEFD at Pol 2"),
        dict(name="d1", dtype="float128", unit="", comment="D-term at Pol 1"),
        dict(name="d2", dtype="float128", unit="", comment="D-term at Pol 2")
    ])

    @property
    def _constructor(self):
        return ANTable

    @property
    def _constructor_sliced(self):
        return ANSeries

    @classmethod
    def make(cls, utc, array, source):
        from pandas import concat
        from astropy.coordinates import GCRS

        # number of time and utc bins
        Nant = len(array.table)

        # compute apparent source coordinates and GST
        skycoord = source.skycoord.transform_to(GCRS(obstime=utc))
        gst = utc.sidereal_time(
            kind="apparent", longitude="greenwich", model="IAU2006A")

        # run loop
        def map_func(iant): return _antable_make_iant(
            iant=iant, utc=utc, gst=gst, array=array, skycoord=skycoord)
        antab = concat([map_func(iant) for iant in range(Nant)])
        return antab


class ANSeries(DataSeries):
    @property
    def _constructor(self):
        return ANSeries

    @property
    def _constructor_expanddim(self):
        return ANTable


# define internal function to compute antenna based tables
def _antable_make_iant(iant, utc, gst, array, skycoord):
    from numpy import exp, cos, sin, tan, arctan2
    from astropy.coordinates import AltAz, EarthLocation
    from ...util.units import DEG, RAD, M, DIMLESS

    # get values
    ant = array.table.loc[iant, :]
    location = EarthLocation(x=ant.x, y=ant.y, z=ant.z, unit=M)
    lon = location.lon
    lat = location.lat
    ra = skycoord.ra
    dec = skycoord.dec

    # compute LST
    lst = utc.sidereal_time(kind="apparent", longitude=lon, model="IAU2006A")

    # compute AZ / alt
    site = AltAz(location=location, obstime=utc)
    altaz = skycoord.transform_to(site)
    el = altaz.alt
    az = altaz.az
    secz = altaz.secz

    # compute sefd
    sefd1 = ant.sefd1 * exp(-ant.tau1*secz)
    sefd2 = ant.sefd2 * exp(-ant.tau2*secz)

    # compute pallactic angle
    H = lst.radian - ra.radian
    cosH = cos(H)
    sinH = sin(H)
    tanlat = tan(lat.radian)
    cosdec = cos(dec.radian)
    sindec = sin(dec.radian)
    par = arctan2(sinH, cosdec*tanlat - sindec*cosH)*RAD

    # compute field rotation angle
    fra = (ant.fr_pa_coeff * DIMLESS) * par
    fra += (ant.fr_el_coeff * DIMLESS) * el
    fra += ant.fr_offset * DEG

    # antab
    antab = ANTable(
        dict(
            antid=iant,
            antname=ant.antname,
            mjd=utc.mjd,
            gst=gst.hour,
            ra=ra.deg,
            dec=dec.deg,
            x=ant.x,
            y=ant.y,
            z=ant.z,
            az=az.deg,
            el=el.deg,
            sefd1=sefd1,
            sefd2=sefd2,
            par=par.to_value(DEG),
            fra=fra.to_value(DEG),
            d1=ant.d1,
            d2=ant.d2
        ),
        columns=ANTable.header.name.to_list()
    )
    antab.convert_format()
    return antab
