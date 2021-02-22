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

# -----------------------------------------------------------------------------
# uvfits loading
# -----------------------------------------------------------------------------


def uvfits2UVData(uvfits, printlevel=0):
    """
    Load an uvfits file. Currently, this function can read only single-source,
    single-frequency-setup, single-array data correctly.

    Args:
        uvfits (string or pyfits.HDUList object): input uvfits data
        printlevel (integer): print some notes. 0: silient 3: maximum level
    Returns:
        uvdata.UVData object
    """
    import astropy.io.fits as pf
    from ....uvdata import UVData

    # check input files
    if isinstance(uvfits, type("")):
        hdulist = pf.open(uvfits)
        closehdu = True
    else:
        hdulist = uvfits
        closehdu = False

    # print HDU info if requested.
    if printlevel > 0:
        hdulist.info()
        print("")

    # load data
    ghdu, antab, fqtab = uvfits2HDUs(hdulist)
    source = uvfits2Source(ghdu=ghdu)
    freq = uvfits2Freq(ghdu=ghdu, antab=antab, fqtab=fqtab)
    array = uvfits2Array(antab=antab)
    vis = uvfits2vis(ghdu=ghdu)

    # close HDU if this is loaded from a file
    if closehdu:
        hdulist.close()

    # create UVData
    uvd = UVData()
    uvd.set_array(array)
    uvd.set_source(source)
    uvd.set_freq(freq)
    uvd.data = vis.copy()
    uvd.set_freq2vis()
    uvd.flags["recalc_uvw"] = True
    uvd.flags["recalc_antab"] = True
    uvd.calc_uvw()

    return uvd


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
    from smili2.util.terminal import warn

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
    from smili2.util import warn
    from astropy.time import Time
    from xarray import DataArray
    from numpy import moveaxis, float64, int32, int64, zeros, where, power
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
    # vis_ghdu = moveaxis(vis_ghdu, 0, -2)  # [if, ch, stokes, data, complex]
    # vis_ghdu = moveaxis(vis_ghdu, 2, 0)  # [if, ch, stokes, data, complex]

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
    antid1 = int64(bl//256)
    antid2 = int64(bl % 256)
    if len(unique(subarray)) > 1:
        warn("Group HDU contains data with 2 or more subarrays.")
        warn("It will likely cause a problem, since SMILI assumes UVFITS for a single subarray.")

    # read polarizations
    stokesids = ghdu.header["CDELT3"] * \
        (arange(Nstokes)+1-ghdu.header["CRPIX3"])+ghdu.header["CRVAL3"]
    stokes = [stokesid2name["%+d" % (stokesid)] for stokesid in stokesids]

    # form a data array
    vis = DataArray(
        vcmp,
        #        dims=["stokes", "if", "ch", "data"],
        dims=["data", "if", "ch", "stokes"],
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
            #            freq=(["if", "ch"], self.freq.get_freqarr()),
            #            chbw=(["if"], self.freq.table["ch_bw"].values)
        )
    )

    return vis


def uvfits2Array(antab):
    """
    Load the rray information from uvfits's AIPS AN table into the SMILI format.

    Args:
        antab (astropy.io.fits.HDU): HDU for AIPS AN table

    Returns:
        array.Array: array information in SMILI format
    """
    from numpy import complex128, asarray, zeros, ones
    from ....util.terminal import warn
    from ....array.array import Array, ArrayTable

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

    d1 = zeros(Nant, dtype=complex128)
    d2 = zeros(Nant, dtype=complex128)

    data = dict(
        antname=antname,
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        sefd1=zeros(Nant),
        sefd2=zeros(Nant),
        tau1=zeros(Nant),
        tau2=zeros(Nant),
        elmin=ones(Nant),
        elmax=ones(Nant)*90.,
        fr_pa_coeff=fr_pa_coeff,
        fr_el_coeff=fr_el_coeff,
        fr_offset=fr_offset,
        d1=d1,
        d2=d2,
        anttype=asarray(["ground" for i in range(Nant)], dtype="U8"),
    )

    table = ArrayTable(
        data=data,
        columns=ArrayTable.header.name.to_list()
    )

    arraydata = Array(name=name, table=table)
    return arraydata


def uvfits2Freq(ghdu, antab, fqtab):
    """
    Load the frequency information from uvfits HDUs into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU
        antab (astropy.io.fits.HDU): HDU for AIPS AN table
        fqtab (astropy.io.fits.HDU): HDU for AIPS FQ table

    Returns:
        freq.Freq object: Loaded frequency table
    """
    from ....util.terminal import warn
    from ....freq.freq import Freq, FreqTable
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
    data = dict(
        if_freq=reffreq + iffreq,
        ch_bw=chbw,
        sideband=sideband,
    )
    freqtable = FreqTable(
        data=data,
        columns=FreqTable.header.name.to_list()
    )
    return Freq(freqtable=freqtable, Nch=Nch, name=name)


def uvfits2Source(ghdu):
    """
    Load the frequency information from uvfits HDUs into the SMILI format.

    Args:
        ghdu (astropy.io.fits.HDU): Group (Primary) HDU

    Returns:
        source.Source object: Loaded source information
    """
    from astropy.coordinates import SkyCoord
    from ....util.units import DEG
    from ....source.source import Source

    # get source name and radec
    srcname = ghdu.header["OBJECT"]
    ra_deg = ghdu.header["CRVAL6"]
    dec_deg = ghdu.header["CRVAL7"]

    # create a skycoord object
    skycoord = SkyCoord(ra=ra_deg*DEG, dec=dec_deg*DEG)

    return Source(name=srcname, skycoord=skycoord)

# -----------------------------------------------------------------------------
# uvfits writing
# -----------------------------------------------------------------------------


def UVData2uvfits(uvd, filename=None, overwrite=True):
    """
    save UVData to uvfits file. If the filename is not given,
    then return HDUList object

    Args:
        filename (str):
            Output uvfits filename
        overwrite (boolean; default=True)
            If True, overwrite when the specified file already exiests.
    Returns:
        astropy.io.fits.HDUList object if filename=None.
    """
    from astropy.io.fits import HDUList

    # create HDUList
    hdulist = []
    # hdulist.append(self._create_ghdu_single())
    # hdulist.append(self._create_fqtab())
    #hdulist += self._create_antab()
    hdulist = HDUList(hdulist)

    # return HDUList or write it to a uvfits file
    if filename is None:
        return hdulist
    else:
        hdulist.writeto(filename, overwrite=overwrite)
        hdulist.close()
