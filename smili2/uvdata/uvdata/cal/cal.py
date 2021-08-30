#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ....util.zarrds import ZarrDataset
from logging import getLogger
logger = getLogger(__name__)


class CalData(ZarrDataset):
    """
    Calibration Data:
    This class is storing the antenna-based calibration table
    in a single dataset
    """
    # Data type name
    name = "Calibration Dataset"

    # Group Name of zarr file
    zarr_group = "calibration"

    def set_time_from_table(self, timetable):
        self.ds.coords["mjd"] = ("time", timetable["mjd"])
        self.ds.coords["dmjd"] = ("time", timetable["dmjd"])
        self.ds.coords["scanid"] = ("time", timetable["scanid"])

    def calc_gst(self):
        from ....util.time import mjd2gst
        gst = mjd2gst(self.ds.mjd.values)
        self.ds.coords["gst"] = ("time", gst)

    def calc_azelfra(self, nproc=-1):
        from ....util.coords import xyz2lstazelfra
        from ....util.time import mjd2utc
        from ....util.units import DEG2RAD, RAD2DEG, RAD2HOUR
        from numpy import asarray

        utc = mjd2utc(self.ds.mjd.values)
        Nant = self.ds.dims["ant"]
        skycoord = self.get_skycoord()

        if nproc < 0:
            outputs = [xyz2lstazelfra(
                x=self.ds.x.values[iant],
                y=self.ds.y.values[iant],
                z=self.ds.z.values[iant],
                utc=utc,
                skycoord=skycoord,
                fr_pa_coeff=self.ds.fr_pa_coeff.values[iant],
                fr_el_coeff=self.ds.fr_el_coeff.values[iant],
                fr_offset=self.ds.fr_offset.values[iant] * DEG2RAD
            ) for iant in range(Nant)]
        else:
            import ray

            ray.init(num_cpus=nproc, ignore_reinit_error=True)
            remote_func = ray.remote(xyz2lstazelfra)
            outputs = [remote_func.remote(
                x=self.ds.x.values[iant],
                y=self.ds.y.values[iant],
                z=self.ds.z.values[iant],
                utc=utc,
                skycoord=skycoord,
                fr_pa_coeff=self.ds.fr_pa_coeff.values[iant],
                fr_el_coeff=self.ds.fr_el_coeff.values[iant],
                fr_offset=self.ds.fr_offset.values[iant] * DEG2RAD
            ) for iant in range(Nant)]
            outputs = ray.get(outputs)

        outputs = asarray(outputs)
        self.ds.coords["lst"] = (["ant", "time"], outputs[:, 0, :]*RAD2HOUR)
        self.ds.coords["az"] = (["ant", "time"], outputs[:, 1, :]*RAD2DEG)
        self.ds.coords["el"] = (["ant", "time"], outputs[:, 2, :]*RAD2DEG)
        self.ds.coords["par"] = (["ant", "time"], outputs[:, 3, :]*RAD2DEG)
        self.ds.coords["fra"] = (["ant", "time"], outputs[:, 4, :]*RAD2DEG)

    def get_skycoord(self):
        from astropy.coordinates import SkyCoord
        from smili2.util.units import DEG

        if "fk" in self.ds.attrs["coordsys"]:
            from astropy.time import Time
            equinox = Time(self.ds.attrs["equinox"], format="jyear")
            srccoord = SkyCoord(
                ra=self.ds.attrs["ra"] * DEG,
                dec=self.ds.attrs["dec"] * DEG,
                equinox=equinox,
                frame=self.ds.attrs["coordsys"]
            )
        elif "gcrs" in self.ds.attrs["coordsys"]:
            from astropy.time import Time
            obstime = Time(self.ds.attrs["equinox"], format="jyear")

            srccoord = SkyCoord(
                ra=self.ds.attrs["ra"] * DEG,
                dec=self.ds.attrs["dec"] * DEG,
                obstime=obstime,
                frame=self.ds.attrs["coordsys"]
            )
        elif "icrs" in self.ds.attrs["coordsys"]:
            srccoord = SkyCoord(
                ra=self.ds.attrs["ra"] * DEG,
                dec=self.ds.attrs["dec"] * DEG,
                frame=self.ds.attrs["coordsys"]
            )
        else:
            msg = "coordsys = %s is not supported" % (
                self.ds.attrs["coordsys"])
            logger.error(msg)
            raise ValueError(msg)

        return srccoord

    @classmethod
    def from_uvd(cls, uvd, timetable=None):
        from xarray import Dataset

        antds = uvd.ant.ds
        freqds = uvd.freq.ds
        srcds = uvd.src.ds

        coords = dict(
            freq=(["if", "ch"], freqds.freq.data)
        )
        for key in antds.coords.keys():
            coords[key] = (antds.coords[key].dims, antds.coords[key].values)

        attrs = dict(
            ra=srcds.attrs["ra"],
            dec=srcds.attrs["dec"],
            equinox=srcds.attrs["equinox"],
            coordsys=srcds.attrs["coordsys"]
        )

        ds = Dataset(
            coords=coords,
            attrs=attrs
        )

        return cls(ds=ds)
