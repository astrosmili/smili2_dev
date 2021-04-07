#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import int64

# Dictionary for Stokes labels and their IDs in UVFITS
stokesid2name = {
    "+1": "I",
    "+2": "Q",
    "+3": "U",
    "+4": "V",
    "-1": "RR",
    "-2": "LL",
    "-3": "RL",
    "-4": "LR",
    "-5": "XX",
    "-6": "YY",
    "-7": "XY",
    "-8": "YX",
}
stokesname2id = {}
for key in stokesid2name.keys():
    stokesname2id[stokesid2name[key]] = int64(key)


def uvfits2UVData(inuvfits, outzarr, scangap=None, nseg=2, printlevel=0):
    """
    Load an uvfits file. Currently, this function can read only single-source,
    single-frequency-setup, single-array data correctly.

    Args:
        uvfits (string or pyfits.HDUList object):
            Input uvfits data
        zarr (string):
            Output zarr file for UVData
        scangap (float or astropy.units.Quantity, optional):
            Minimal time seperation between scans.
            If not specfied, this will be guessed from data segmentation (see nseg).
            If a float value is specified, its unit is assumuted to be in seconds.
            Defaults to None.
        nseg (float, optional):
            If scangap is None, the minimal time seperation between scans
            will be set to nseg * minimal_data_segementation_time.
            Defaults to 2.
        printlevel (integer, optional):
            print some notes. 0: silient 3: maximum level
    Returns:
        uvdata.UVData object
    """
    import astropy.io.fits as pf
    import os
    from .zarr import zarr2UVData
    import zarr

    # check input files
    if isinstance(inuvfits, type("")):
        hdulist = pf.open(inuvfits)
        closehdu = True
    else:
        hdulist = inuvfits
        closehdu = False

    # print HDU info if requested.
    if printlevel > 0:
        hdulist.info()
        print("")

    # load data
    ghdu, antab, fqtab = uvfits2HDUs(hdulist)

    # create zarr file
    z = zarr.open(outzarr, mode="w")

    # Load info from HDU
    #   Frequency
    uvfits2freq(ghdu=ghdu, antab=antab, fqtab=fqtab).to_zarr(outzarr)
    del fqtab
    #   Antenna
    uvfits2ant(antab=antab).to_zarr(outzarr)
    del antab
    #   Source
    uvfits2src(ghdu=ghdu).to_zarr(outzarr)
    #   Visibilities
    vistab = uvfits2vistab(ghdu=ghdu)
    del ghdu

    # Detect scans and save visibilities and scaninfo to zarr file
    vistab.set_scan(scangap=scangap, nseg=2)
    vistab.to_zarr(outzarr)
    vistab.gen_scandata().to_zarr(outzarr)

    # close HDU if this is loaded from a file
    if closehdu:
        hdulist.close()

    # open zarr file and return it
    return zarr2UVData(inzarr=outzarr)


def uvfits2HDUs(hdulist):
    """
    Read HDUList, and get PrimaryHDU & HDUS for AIPS AN/FQ Tables

    Args:
        hdulist (astropy.io.fits.HDUList): hdulist

    Returns:
        Group HDU
        HDU for AIPS AN Table
        HDU for AIPS FQ Table
    """
    from ....util.terminal import warn

    hduinfo = hdulist.info(output=False)
    Nhdu = len(hduinfo)

    fqtab = None
    antab = None
    ghdu = None
    for ihdu in range(Nhdu):
        hduname = hduinfo[ihdu][1]
        if "PRIMARY" in hduname.upper():
            if ghdu is not None:
                warn("This UVFITS has more than two Primary HDUs.")
                warn("The later one will be taken.")
            ghdu = hdulist[ihdu]
        elif "FQ" in hduname.upper():
            if fqtab is not None:
                warn("This UVFITS has more than two AIPS FQ tables.")
                warn("The later one will be taken.")
            fqtab = hdulist[ihdu]
        elif "AN" in hduname.upper():
            if antab is not None:
                warn("This UVFITS has more than two AIPS AN tables.")
                warn("The later one will be taken.")
            antab = hdulist[ihdu]

    return ghdu, antab, fqtab


def uvfits2vistab(ghdu):
    """
    Load the array information from uvfits's AIPS AN table into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU

    Returns:
        VisData: complex visibility in SMILI format
    """
    from ....util import warn
    from ..vis.vistab import VisTable
    from astropy.time import Time
    from xarray import Dataset
    from numpy import float64, int32, int64, zeros, where, power
    from numpy import abs, sign, isinf, isnan, finfo, unique, modf, arange, min, diff

    # read visibilities
    #    uvfits's original dimension is [data,dec,ra,if,ch,stokes,complex]
    #    first, we reorganize the array to [stokes,if,ch]
    Ndata, Ndec, Nra, dammy, dammy, Nstokes, dammy = ghdu.data.data.shape
    del dammy
    if Nra > 1 or Ndec > 1:
        warn("GroupHDU has more than single coordinates (Nra, Ndec)=(%d, %d)." % (Nra, Ndec))
        warn("We will pick up only the first one.")
    vis_ghdu = ghdu.data.data[:, 0, 0, :]  # to [data,if,ch,stokes,complex]

    # get visibilities, errors, and flag (flagged, removed,)
    vcmp = float64(vis_ghdu[:, :, :, :, 0]) + 1j * \
        float64(vis_ghdu[:, :, :, :, 1])
    sigma = float64(power(abs(vis_ghdu[:, :, :, :, 2]), -0.5))
    flag = int32(sign(vis_ghdu[:, :, :, :, 2]))

    # check sigma
    idx = where(isinf(sigma))
    sigma[idx] = 0
    flag[idx] = 0

    idx = where(isnan(sigma))
    sigma[idx] = 0
    flag[idx] = 0

    idx = where(sigma < finfo(float64).eps)
    sigma[idx] = 0
    flag[idx] = 0

    # Read Random Parameters
    paridxes = [None for i in range(9)]
    parnames = ghdu.data.parnames
    Npar = len(parnames)
    jd1 = zeros(Ndata)
    jd2 = zeros(Ndata)
    for i in range(Npar):
        parname = parnames[i]
        if "UU" in parname:
            paridxes[0] = i+1
            usec = float64(ghdu.data.par(i))
        if "VV" in parname:
            paridxes[1] = i+1
            vsec = float64(ghdu.data.par(i))
        if "WW" in parname:
            paridxes[2] = i+1
            wsec = float64(ghdu.data.par(i))
        if "DATE" in parname:
            if paridxes[3] is None:
                paridxes[3] = i+1
                jd1 = float64(ghdu.data.par(i))
            elif paridxes[4] is None:
                paridxes[4] = i+1
                jd2 = float64(ghdu.data.par(i))
            else:
                errmsg = "Random Parameters have too many 'DATE' columns."
                raise ValueError(errmsg)
        if "BASELINE" in parname:
            paridxes[5] = i+1
            bl = float64(ghdu.data.par(i))
        if "SOURCE" in parname:
            paridxes[6] = i+1
            srcid = int32(ghdu.data.par(i))
        if "INTTIM" in parname:
            paridxes[7] = i+1
            inttim = float64(ghdu.data.par(i))
        if "FREQSEL" in parname:
            paridxes[8] = i+1
            freqsel = int32(ghdu.data.par(i))

    # convert JD to MJD
    mjd = Time(jd1, jd2, format="jd").mjd

    # warn if it is an apparently multi source file
    if paridxes[6] is not None:
        if len(unique(srcid)) > 1:
            warn("Group HDU contains data on more than a single source.")
            warn(
                "It will likely cause a problem since SMILI assumes a singlesource UVFITS.")

    # Integration time in the unit of day
    if paridxes[7] is None:
        warn("Group HDU do not have a random parameter for the integration time.")
        warn("It will be estimated with a minimal time interval of data.")
        dmjd = min(abs(diff(unique(mjd))))
    else:
        dmjd = inttim/86400

    # warn if data are apparently with multi IF setups
    if paridxes[8] is not None:
        if len(unique(freqsel)) > 1:
            warn("Group HDU contains data on more than a frequency setup.")
            warn(
                "It will likely cause a problem since SMILI assumes a UVFITS with a single setup.")

    # antenna ID
    subarray, bl = modf(bl)
    subarray = int64(100*(subarray)+1)
    antid1 = int64(bl//256)-1
    antid2 = int64(bl % 256)-1
    if len(unique(subarray)) > 1:
        warn("Group HDU contains data with 2 or more subarrays.")
        warn("It will likely cause a problem, since SMILI assumes UVFITS for a single subarray.")

    # read polarizations
    stokesids = ghdu.header["CDELT3"] * \
        (arange(Nstokes)+1-ghdu.header["CRPIX3"])+ghdu.header["CRVAL3"]
    stokes = [stokesid2name["%+d" % (stokesid)] for stokesid in stokesids]

    # form a data array
    ds = Dataset(
        data_vars=dict(
            vis=(["data", "if", "ch", "stokes"], vcmp)
        ),
        coords=dict(
            mjd=("data", mjd),
            dmjd=("data", dmjd),
            usec=("data", usec),
            vsec=("data", vsec),
            wsec=("data", wsec),
            antid1=("data", antid1),
            antid2=("data", antid2),
            flag=(["data", "if", "ch", "stokes"], flag),
            sigma=(["data", "if", "ch", "stokes"], sigma),
            stokes=(["stokes"], stokes),
        )
    )
    return VisTable(ds=ds.sortby(["mjd", "antid1", "antid2"]))


def uvfits2ant(antab):
    """
    Load the rray information from uvfits's AIPS AN table into the SMILI format.

    Args:
        antab (astropy.io.fits.HDU): HDU for AIPS AN table

    Returns:
        AntData: array information in SMILI format
    """
    from numpy import asarray, zeros, ones
    from ....util.terminal import warn
    from ..ant.ant import AntData
    from xarray import Dataset

    # The array name
    name = antab.header["ARRNAM"]

    # Number of Antenna
    Nant = len(antab.data)

    # Anteanna Name
    antname = antab.data["ANNAME"].tolist()

    # XYZ Coordinates
    xyz = antab.data["STABXYZ"]

    # Parse Field Rotation Information
    #   See AIPS MEMO 117
    #      0: ALT-AZ, 1: Eq, 2: Orbit, 3: X-Y, 4: Naismith-R, 5: Naismith-L
    #      6: Manual
    mntsta = antab.data["MNTSTA"]
    fr_pa_coeff = ones(Nant)
    fr_el_coeff = zeros(Nant)
    fr_offset = zeros(Nant)
    for i in range(Nant):
        if mntsta[i] == 0:  # azel
            fr_pa_coeff[i] = 1
            fr_el_coeff[i] = 0
        elif mntsta[i] == 1:  # Equatorial
            fr_pa_coeff[i] = 0
            fr_el_coeff[i] = 0
        elif mntsta[i] == 4:  # Nasmyth-R
            fr_pa_coeff[i] = 1
            fr_el_coeff[i] = 1
        elif mntsta[i] == 5:  # Nasmyth-L
            fr_pa_coeff[i] = 1
            fr_el_coeff[i] = -1
        else:
            warn("[WARNING] MNTSTA %d at Station %s is not supported currently." % (
                mntsta[i], antname[i]))

    # assume all of them are ground array
    anttype = asarray(["g" for i in range(Nant)], dtype="U8")

    ant = Dataset(
        coords=dict(
            antname=("ant", antname),
            x=("ant", xyz[:, 0]),
            y=("ant", xyz[:, 1]),
            z=("ant", xyz[:, 2]),
            fr_pa_coeff=("ant", fr_pa_coeff),
            fr_el_coeff=("ant", fr_el_coeff),
            fr_offset=("ant", fr_offset),
            anttype=("ant", anttype),
        ),
        attrs=dict(
            name=name,
        ),
    )
    return AntData(ant)


def uvfits2freq(ghdu, antab, fqtab):
    """
    Load the frequency information from uvfits HDUs into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU
        antab (astropy.io.fits.HDU): HDU for AIPS AN table
        fqtab (astropy.io.fits.HDU): HDU for AIPS FQ table

    Returns:
        FreqData: Loaded frequency table
    """
    from ....util.terminal import warn
    from ..freq import FreqData
    from xarray import Dataset
    from numpy import float64

    # read meta data from antenna table
    reffreq = antab.header["FREQ"]  # reference frequency in GHz
    # use array name because uvfits doesn't have such meta data
    name = antab.header["ARRNAM"]

    # get number of channels
    dammy, dammy, dammy, Nif, Nch, dammy, dammy = ghdu.data.data.shape
    del dammy

    # read data from frequency table
    nfrqsel = len(fqtab.data["FRQSEL"])
    if nfrqsel > 1:
        warn("Input FQ Tables have more than single FRQSEL. We only handle a uvfits with single FRQSEL.")
    iffreq = float64(fqtab.data["IF FREQ"][0])
    chbw = float64(fqtab.data["CH WIDTH"][0])
    sideband = float64(fqtab.data["SIDEBAND"][0])

    # check the consistency between the number of if in FQ Table and GroupHDU
    if len(iffreq) != Nif:
        raise ValueError(
            "Group HDU has %d IFs, which is inconsistent with FQ table with %d IFs" % (
                Nif, len(iffreq))
        )

    # Make FreqTable
    dataset = Dataset(
        coords=dict(
            if_freq=("if", reffreq+iffreq),
            ch_bw=("if", chbw),
            sideband=("if", sideband)
        ),
        attrs=dict(
            name=name,
            Nch=Nch,
        )
    )
    freq = FreqData(dataset)
    freq.recalc_freq()

    return freq


def uvfits2src(ghdu):
    """
    Load the source information from uvfits HDUs into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU

    Returns:
        SrcData: Loaded frequency table
    """
    from ..src.src import SrcData
    from xarray import Dataset

    # source info
    srcname = ghdu.header["OBJECT"]
    ra = ghdu.header["CRVAL6"]
    dec = ghdu.header["CRVAL7"]

    src = Dataset(
        attrs=dict(
            name=srcname,
            ra=ra,
            dec=dec
        ),
    )

    return SrcData(src)
