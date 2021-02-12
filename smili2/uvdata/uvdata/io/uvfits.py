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


def uvfits2vis(ghdu):
    """
    Load the rray information from uvfits's AIPS AN table into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU

    Returns:
        xarray.DataArray: complex visibility in SMILI format
    """
    from ....util import warn
    from .. import VisData
    from astropy.time import Time
    from xarray import Dataset
    from numpy import moveaxis, float64, int32, int64, zeros, where, power, deg2rad
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

    # source info
    srcname = ghdu.header["OBJECT"]
    ra = ghdu.header["CRVAL6"]
    dec = ghdu.header["CRVAL7"]

    # form a data array
    vis = Dataset(
        data_vars=dict(
            vis=(["data", "if", "ch", "stokes"], vcmp),
            flag=(["data", "if", "ch", "stokes"], flag),
            sigma=(["data", "if", "ch", "stokes"], sigma),
        ),
        coords=dict(
            mjd=("data", mjd),
            dmjd=("data", dmjd),
            usec=("data", usec),
            vsec=("data", vsec),
            wsec=("data", wsec),
            antid1=("data", antid1),
            antid2=("data", antid2),
            stokes=(["stokes"], stokes),
        ),
        attrs=dict(
            srcname=srcname,
            ra=ra,
            dec=dec,
        )
    )

    return VisData(vis)
