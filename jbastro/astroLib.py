import numpy as np
import astropy
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import scipy.ndimage

from jbastro.great_circle_dist import dist_radec_fast
from jbastro.astrolibsimple import roundTo, dm2d, d2dm, sexconvert, dec2sex, sex2dec

SPEED_OF_LIGHT = 299792458.0
M2FS_FOV_DEG = 29.0 / 60

from astropy.io import fits
# from cosmics.cosmics import cosmicsimage
from astroscrappy import detect_cosmics


def do_kde(y, x=None, scipykde=False, norm=False):
    if scipykde:
        from scipy.stats import gaussian_kde as gkde
        pdf = gkde(y)(x)
        bw = None
    else:
        from .kde.kde import kde
        bw, x, pdf = kde(y)

    dx = x[1] - x[0]

    peaki = pdf.argmax()
    peakx = x[peaki]
    nval = (pdf * dx).sum()
    try:
        ret = []
        for i in range(x.size):
            if i <= peaki: ret.append((pdf[:i + 1] * dx).sum() / nval)
            if i > peaki: ret.append((pdf[i - 1:] * dx).sum() / nval)
        ret = np.array(ret)
        foo = x[np.abs(np.array(ret) - .16).argsort()]
        msig, psig = foo[foo < peakx][0], foo[foo > peakx][0]  # 1sigma values
    except IndexError:
        msig, psig, ret = np.nan, np.nan, None
    if norm: pdf /= pdf.max()

    return x, pdf, (msig, psig), {'ppf': ret, 'bw': bw, 'norm': nval}


def crreject(im, dialate=False, **cosmics_settings):
    """Give a seqno or a path to a quad if file set"""
    # def_cosmic_settings = {'sigclip': 6.0, 'sigfrac': 0.5,
    #                        'objlim': 1.4, 'iter': 7, 'readnoise': 0.0,
    #                        'gain': 1.0, 'satlevel': .95 * (2 ** 16)}

    def_cosmic_settings = {'sigclip': 6.0, 'sigfrac': 0.5,
                           'objlim': 1.4, 'niter': 7, 'readnoise': 0.0,
                           'gain': 1.0, 'satlevel': .95 * (2 ** 16),
                           'sepmed': True, 'pssl': 0.0, 'fsmode': 'median',
                           'psfmodel': 'gaussy', 'psffwhm': 2.5, 'psfsize': 7,
                           'psfk': None, 'psfbeta': 4.765, 'verbose': True}

    for k in def_cosmic_settings:
        if k not in cosmics_settings:
            cosmics_settings[k] = def_cosmic_settings[k]

    # cosmics_settings['niter'] = cosmics_settings.pop('iter')
    # mask, _clean = detect_cosmics(im, inmask=None, float sigclip=4.5, float sigfrac=0.3,
    #                                float objlim=5.0, float gain=1.0, float readnoise=6.5,
    #                                float satlevel=65536.0, float pssl=0.0, niter=cosmic_iter,
    #                                sepmed=True, cleantype='meanmask', fsmode='median',
    #                                psfmodel='gauss', float psffwhm=2.5, int psfsize=7,
    #                                psfk=None, float psfbeta=4.765, verbose=False)
    mask, _ = detect_cosmics(im, **cosmics_settings)
    mask = mask.astype(np.bool)
    if dialate:
        mask = scipy.ndimage.morphology.binary_dilation(mask, structure=np.ones((3, 3)),
                                                        iterations=1, mask=None, output=None,
                                                        border_value=0, origin=0, brute_force=False)

    return mask

    # cosmic_iter = cosmics_settings.pop('iter')
    # c = cosmicsimage(im, **cosmics_settings)
    # c.run(maxiter=cosmic_iter)
    # return c.mask.astype(np.uint8)




def cycscatter(*args, **kwargs):
    import matplotlib.pyplot as plt
    """Make a cyclic scatter plot"""
    x = args[0].copy()
    if 'set_xlim' in kwargs:
        set_xlim = kwargs.pop('set_xlim')
    else:
        set_xlim = False
    if 'forcecyc' in kwargs:
        force = kwargs.pop('forcecyc')
    else:
        force = False
    if force or (len(np.where(x < 100)[0]) > 0 and len(np.where(x > 300)[0]) > 0):
        if x.shape == ():
            x = x + 360.
        else:
            x[x < 100] = x[x < 100] + 360.
        from matplotlib.ticker import FuncFormatter
        @FuncFormatter
        def wrapticks(x, pos):
            return "{:.1f}".format(x % 360)

        # tmp=plt.gca().xaxis.get_major_formatter()
        plt.gca().xaxis.set_major_formatter(wrapticks)
        plt.scatter(x, *args[1:], **kwargs)
        if set_xlim:
            plt.gca().set_xlim((min(x), max(x)))
        # plt.gca().xaxis.set_major_formatter(tmp)
        return True
    else:
        plt.scatter(*args, **kwargs)
        return False


def getIsochrone(age, color='VJ'):
    """Takes age in Myr color=vj or bv"""
    if age < 100:
        grid_age = roundTo(age, 10)
    elif age < 500:
        grid_age = roundTo(age, 50)
    elif age < 1000:
        grid_age = roundTo(age, 100)
    elif age < 3500:
        grid_age = roundTo(age, 250)
    elif age < 3700:
        grid_age = roundTo(age, 100)
    elif age < 3800:
        grid_age = roundTo(age, 50)
    elif age < 4000:
        grid_age = roundTo(age, 100)
    elif age < 16250:
        grid_age = roundTo(age, 500)
    else:
        raise ValueError("No Isochrone for {0}".format(age))
    isofile = 'isoz22s_c03hbs/wzsunysuns.t6{grid_age:05}_c03hbs'

    # Load V & V-J (col 9) NB B-v is col 6
    if color.lower() == 'vj':
        return np.loadtxt(isofile.format(grid_age=grid_age), comments='#',
                          unpack=True, usecols=(4, 9))
    elif color.lower() == 'bv':
        return np.loadtxt(isofile.format(grid_age=grid_age), comments='#',
                          unpack=True, usecols=(4, 6))


def RtoV(Rabs):
    """
    Rabs -> Vabs F5-K5
    Interpolates based on values for F5,G5, K3, & K5 MS dwarfs
    Dom=[3.17,6.53]
    """
    ygrid = np.array([3.5, 5.1, 6.8, 7.5])
    xgrid = np.array([3.5 - .33, 5.1 - .47, 6.8 - .8, 7.5 - .97])
    return np.interp(Rabs, xgrid, ygrid)


# def expTime(paramTriple):
#    """ (SNR, mag, expTime) """
#    snr, mag, t = paramTriple
#     1/3 e/s/pix @18.9
#     1 c/s/A @18.7 V band (mgb filter) #Per mario November run night 2
#    
#    count_rate= zp* 10**((mag-18.9)/-2.5)
#    
#    if snr is None:
#        snr = 175./np.sqrt(3) * 10**(.2*(13.5-mag))*np.sqrt(t)
#    if mag is None:
#        if snr <=0:
#            mag=float('inf')
#        else:
#            mag =13.5 - np.log10(np.sqrt(3)*snr/np.sqrt(t)/175.)/.2
#    if t is None:
#        t = (snr / (175./np.sqrt(3) * 10**(.2*(13.5-mag))))**2.
#    return (snr, mag, t)

# see seeing_light_loss.nb, values assume a .1" miscentration, though
# that effect is very weak
_magloss = np.array(((0.0, 0.0), (0.2, 0.0), (.4, .05), (.484, .1), (.6, .2),
                     (.688, .3), (.776, .4), (.84, .5), (.927, .6), (1.01, .7),
                     (1.085, .8), (1.16, .9), (1.239, 1.0), (1.619, 1.5),
                     (2.189, 2.0))).T

_magloss_spline = IUS(_magloss[0], _magloss[1])


def seeing_mag_loss(seeing):
    if seeing > _magloss[0].max() or seeing < _magloss[0].min():
        raise ValueError('Seeing out of bounds')
    return _magloss_spline(seeing)


def expTime(paramTriple, seeing=1.0, zero_point=18.3, slit=45, fibtpt=.6,
            aperpix=0.05, teltpt=.8):
    """ (SNR, mag, expTime) 
    zero_point is 1 e/s/A with wide open slit
    36% of light gets through with 45 um slit
    
    """
    snr, mag, t = paramTriple

    slit_tpt = {45: .36, 180: 1.0}
    slit_tpt = {45: .34, 58: .45, 75: .57, 180: 1.0}
    if slit not in slit_tpt.keys(): raise ValueError('Slit not supported.')

    zpcor = slit_tpt[slit] * aperpix * fibtpt * teltpt

    zpm = zero_point
    zpm -= seeing_mag_loss(seeing)

    if snr is None:
        snr = np.sqrt(zpcor * 10 ** ((mag - zpm) / -2.5) * 3600 * t)
    if mag is None:
        if snr <= 0:
            mag = float('inf')
        else:
            mag = np.log10((snr ** 2) / zpcor / 3600 / t) * -2.5 + zpm
    if t is None:
        t = (snr ** 2) / (zpcor * 10 ** ((mag - zpm) / -2.5) * 3600)
    return (snr, mag, t)


def estMag(sptype, band='R'):
    """V or R (default) abs mag for a B0-M0 star """
    try:
        mult = ['B', 'A', 'F', 'G', 'K', 'M'].index(sptype[0].upper())
    except ValueError:
        raise ValueError('Invalid spectral type ({})'.format(sptype))

    x = np.array([0, 1, 2, 3, 4,
                  5, 8, 10, 12, 15,
                  17, 20, 22, 25, 28,
                  30, 32, 35, 38, 40,
                  42, 43, 45, 47, 50])
    VmR = np.array([-.13, -.11, -.1, -.08, -.07,
                    -.06, -.02, 0.02, 0.08, 0.16,
                    0.19, 0.3, 0.35, 0.4, 0.47,
                    0.5, 0.53, 0.54, 0.58, 0.64,
                    0.74, 0.81, 0.99, 1.15, 1.28])
    V = np.array([-3.3, -2.9, -2.5, -2.0, -1.5,
                  -1.1, 0.0, 0.7, 1.3, 1.9,
                  2.3, 2.7, 3.0, 3.5, 4.0,
                  4.4, 4.7, 5.1, 5.6, 6.0,
                  6.5, 6.8, 7.5, 8.0, 8.8])
    if band == 'R':
        return np.interp(int(sptype[1]) + mult * 10, x, V - VmR)
    elif band == 'V':
        return np.interp(int(sptype[1]) + mult * 10, x, V)


def estBV(sptype):
    """ 
    B-V value for a B0-M0 star
    
    Appendix B, Gray
    """
    try:
        mult = ['B', 'A', 'F', 'G', 'K', 'M'].index(sptype[0].upper())
    except ValueError:
        raise ValueError('Invalid spectral type ({})'.format(sptype))
    return np.interp(int(sptype[1]) + mult * 10,
                     np.array([0, 1, 2, 3, 4,
                               5, 8, 10, 12, 15,
                               17, 20, 22, 25, 28,
                               30, 32, 35, 38, 40,
                               42, 43, 45, 47, 50]),
                     np.array([-.29, -.26, -.24, -.21, -.18,
                               -.16, -.10, 0.0, 0.06, 0.14,
                               0.19, 0.31, 0.36, 0.44, 0.53,
                               0.59, 0.63, 0.68, 0.74, 0.82,
                               0.92, 0.96, 1.15, 1.30, 1.41]))


def estSpType(absmag, dm=None, band='V'):
    """Spectral type for Mag, mag if dm=modulus, V(default) or R"""
    if dm != None:
        absmag = absmag - dm
    if band == 'V':
        type = int(round(np.interp(absmag, np.array([3.5, 5.1, 6.8, 7.5]),
                                   np.array([0, 10, 18, 20]))))
    elif band == 'R':
        type = int(round(np.interp(absmag,
                                   np.array([3.5 - .33, 5.1 - .47, 6.8 - .8, 7.5 - .97]),
                                   np.array([0, 10, 18, 20]))))
    if type < 5:
        type = 'F' + str((type % 10) + 5)[-1]
    elif type < 15:
        type = 'G' + str((type % 10) + 5)[-1]
    elif type < 21:
        type = 'K' + str((type % 10) + 5)[-1]
    return type


def massLuminosity(mass):
    if mass < .43:
        k = .23
        a = 2.3
    elif mass < 2:
        k = 1.0
        a = 4
    elif mass < 20:
        k = 1.5
        a = 3.5
    else:
        raise ValueError("Mass too large")
    if mass < 0.08:
        alpha = .3
    if mass < 0.5:
        alpha = 1.3
    else:
        alpha = 2.3
    beta = alpha / a / 2.5
    omega = k ** (alpha / a)
    j = (10.0 ** (beta * c1) - 10.0 ** (beta * c2))
    mbol = (-1.0 / b) * np.log10(N * b * np.log(10) / o / j)


def VtoR(Vabs):
    xgrid = np.array([3.5, 5.1, 6.8, 7.5])
    ygrid = np.array([3.5 - .33, 5.1 - .47, 6.8 - .8, 7.5 - .97])
    return np.interp(Vabs, xgrid, ygrid)


def rvPrecision(snr, sptype='K5'):
    xgrid = np.array([15, 35, 55, 75, 95, 115, 175, 235, 295, 315, 375])
    ygrid = np.array([236, 101, 64, 47, 37, 31, 20, 15, 12, 11, 9])
    sigma = int(round(np.interp(snr, xgrid, ygrid)))
    if 'G' in sptype:
        sigma *= 1.065  # emperic estimate from IDL numbers
    elif 'F' in sptype:
        sigma *= 1.59
    return sigma


#
# S/N      Atm          F5          K5          G5
# 15       179         375         236         251
# 35        77         161         101         108
# 55        49         102          64          68
# 75        36          75          47          50
# 95        28          59          37          40
# 115       23          49          31          33
# 175       15          32          20          22
# 235       11          24          15          16
# 295        9          19          12          13
# 315        9          18          11          12
# 375        7          15           9          10

def obsTimes(snr, dm):
    # F5
    Rmag = dm + 3.5 - .33
    f5t = expTime((snr, Rmag, None))[2]
    # G5
    Rmag = dm + 5.1 - .47
    g5t = expTime((snr, Rmag, None))[2]
    # K3
    Rmag = dm + 6.8 - .8
    k3t = expTime((snr, Rmag, None))[2]
    # K5
    Rmag = dm + 7.5 - .97
    k5t = expTime((snr, Rmag, None))[2]
    # 3hr depth
    mr = expTime((snr, None, 3))[1] - dm
    print("Time to {0} for F5:{1:.2f} G5:{2:.2f}" +
          "K3:{k3:.2f} K5:{3:.2f}. 3h to MR={4:.1f}").format(
        snr, f5t, g5t, k5t, mr, k3=k3t)


def where2bool(length, whereTrue):
    out = np.zeros(length, dtype=np.bool)
    out[whereTrue] = True
    return out


def in_field(coord, stars, fov=M2FS_FOV_DEG, square=False, mask=False):
    """
        Return indices of stars which are in the field
        
        stars may be array of ras & decs, a list of two arrays or an array of
        records with the RAJ2000 & DEJ2000 keys
        
        if filter is set returns a boolean mask
        """

    if type(coord) in [astropy.io.fits.fitsrec.FITS_rec,
                       astropy.io.fits.fitsrec.FITS_record]:
        ctr = (coord['RAJ2000'], coord['DEJ2000'])
    elif type(coord) == astropy.coordinates.builtin_systems.ICRSCoordinates:
        ctr = (coord.ra.degrees, coord.dec.degrees)
    else:
        ctr = coord

    if type(stars) == astropy.io.fits.fitsrec.FITS_rec:
        ras = stars['RAJ2000']
        des = stars['DEJ2000']
    else:
        ras = stars[0]
        des = stars[1]
    try:
        # Extract a square of stars surrounding the field
        ra_hwid = fov / np.cos(ctr[1] * np.pi / 180.0) / 2.0
        de_hwid = fov / 2.0

        ra_min = ctr[0] - ra_hwid
        ra_max = ctr[0] + ra_hwid
        de_min = ctr[1] - de_hwid
        de_max = ctr[1] + de_hwid
        if ra_min > ra_max:
            ra_cut = ((ras > ra_min) |
                      (ras < ra_max))
        else:
            ra_cut = ((ras > ra_min) &
                      (ras < ra_max))

        de_cut = ((des > de_min) & (des < de_max))

        cand = ra_cut & de_cut

        if not square:
            # Now make sure they are all within the field
            sep = dist_radec_fast(ctr[0], ctr[1],
                                  ras[cand], des[cand],
                                  method='Haversine', unit='deg',
                                  scale=fov / 2.0)
            cand[cand] = sep < (fov / 2.0)
    except Exception:
        import ipdb;
        ipdb.set_trace()
    if mask:
        return cand
    else:
        return np.where(cand)[0]


def gauss2dmodel(xw, yw, amp, xo, yo, sigma_x, sigma_y, covar, offset):
    xo = float(xo)
    yo = float(yo)

    x = np.arange(xw + 1, dtype=np.float) - xw / 2 + xo
    y = np.arange(yw + 1, dtype=np.float) - yw / 2 + yo
    x, y = np.meshgrid(x, y)

    rho = covar / (sigma_x * sigma_y)

    z = ((x - xo) / sigma_x) ** 2 - 2 * rho * (x - xo) * (y - yo) / (sigma_x / sigma_y) + ((y - yo) / sigma_y) ** 2

    g = amp * np.exp(-z / (2 * (1 - rho ** 2))) + offset

    return g


# def gaussfit(xdata, ydata):
#    """ fit a gaussian + quadratic to data, doesn't work"""
#    def gauss_quad(x, a0, a1, a2, a3, a4, a5):
#        z = (x - a1) / a2
#        y = a0 * np.exp(-z**2 / a2) + a3 + a4 * x + a5 * x**2
#        return y
#
#    from scipy.optimize import curve_fit
#    parameters, covariance = curve_fit(gauss_quad, xdata, ydata)
#
#    return (parameters, gauss_quad(xdata, *parameters))


def gaussfit(xdata, ydata, p0=None, sig=1, ret_e=False):
    """ fit a gaussian + constant to data, p0=amp,ctr,sig,offset"""

    def gauss(x, a0, a1, a2, a3):
        z = (x - a1) / a2
        y = a0 * np.exp(-z ** 2 / 2) + a3
        return y

    if p0 is None:
        p0 = (ydata.max() - ydata.min(), xdata[ydata.argmax()], sig, ydata.min())

    from scipy.optimize import curve_fit
    parameters, covariance = curve_fit(gauss, xdata, ydata, p0=p0)

    if ret_e:
        return parameters, gauss(xdata, *parameters), np.sqrt(covariance.diagonal())
    else:
        return (parameters, gauss(xdata, *parameters))


def gauss2D(xy, amp, xo, yo, sigma_x, sigma_y, covar, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)

    rho = covar / (sigma_x * sigma_y)

    z = ((x - xo) / sigma_x) ** 2 - 2 * rho * (x - xo) * (y - yo) / (sigma_x / sigma_y) + ((y - yo) / sigma_y) ** 2

    g = amp * np.exp(-z / (2 * (1 - rho ** 2))) + offset

    return g


def gaussfit2D(im, initialp, ftol=1e-5, maxfev=5000, retcov=False):
    """initalp = (amp, x0,y0, sx, yx, covar, offset)"""

    def g2d(xy, amp, xo, yo, sigma_x, sigma_y, covar, offset):
        return gauss2D(xy, amp, xo, yo,
                       sigma_x, sigma_y, covar, offset).ravel()

    x = np.arange(im.shape[0], dtype=np.float)
    y = np.arange(im.shape[1], dtype=np.float)
    x, y = np.meshgrid(x, y)

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(g2d, (x, y), im.ravel(), p0=initialp,
                           ftol=ftol, maxfev=maxfev)

    model = gauss2D((x, y), *popt)

    if retcov:
        return model, popt, pcov
    else:
        return model, popt


def aniscatter(x, y, **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    numframes = len(x)

    try:
        interval = kwargs.pop('interval')
    except Exception:
        interval = 200

    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'

    fig = plt.gcf()
    line, = plt.plot(x[[]], y[[]], linestyle='none', **kwargs)

    def update_plot(i):
        line.set_data(x[:i], y[:i])
        return line,

    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  interval=interval, repeat=False)
    plt.show()
    # display_animation(ani)


def anim_to_html(anim):
    VIDEO_TAG = """<video controls>
        <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
        Your browser does not support the video tag.
        </video>"""
    from tempfile import NamedTemporaryFile
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)


def display_animation(anim):
    import matplotlib.pyplot as plt
    plt.close(anim._fig)
    from IPython.display import HTML
    return HTML(anim_to_html(anim))


def extract_key_from_FITSrec_list(targetThing, key):
    """
    Return a list of two arrays, RAs & DECs. Creates copy of coords.
    """
    if type(targetThing) == astropy.io.fits.fitsrec.FITS_rec:
        out = targetThing[key].copy()
    else:
        t_lens = map(len, targetThing)
        n = sum(t_lens)
        out = np.zeros(n, dtype=targetThing[0][key].dtype)
        for i, arr in enumerate(targetThing):
            ndx = sum(t_lens[0:i])
            out[ndx:ndx + len(arr)] = arr[key]
    return out


def pltradec(thing, clear=False, lw=0, m='.', c='k', fig=None):
    import matplotlib.pyplot as plt
    """thing must have keys RAJ2000 and DEJ2000 """
    # global clusters
    if fig != None:
        plt.figure(fig)
    if clear:
        plt.clf()
    plt.scatter(thing['RAJ2000'], thing['DEJ2000'], c=c, marker=m, lw=lw)
    plt.ylabel('Dec')
    plt.xlabel('RA')
    plt.show()


def baryvel_los(obstime, coords, observatory_loc, sun=False,
                heliocentric=False, aORg='g'):
    """
    vvec (output vector(4))
    Various projections of the barycentric velocity
    correction, expressed in km/sec. The four elements in the vector are radial, tangential,
    right ascension, and declination projections respectively. Add vvec(0) to the observed velocity
    scale to shift it to the barycenter.
    
    sun=True to get shift of scattered light from sun. Coords don't matter
    """
    from PyAstronomy.pyasl import baryvel
    from astropy.coordinates import Longitude, ICRS
    import astropy.units as u
    from astropy.time import Time
    import os

    OBSERVATORY_COORD = {
        'lick3': {
            'lat': 0.651734547,  # +37 20 29.9
            'lon': 2.123019229,  # 121 38 24.15
            'ht': 1283.},
        'cfht': {
            'lat': 0.346030917,  # +19 49 34
            'lon': 2.713492477,  # 155 28 18
            'ht': 4198.},
        'kp': {
            'lat': 0.557865407,  # +31 57.8 (1991 Almanac)
            'lon': 1.947787445,  # 111 36.0
            'ht': 2120.},
        'mcd': {  # McDonald (Fort Davis)
            'lat': 0.5353215959,  # +30 40.3 (1995 Almanac)
            'lon': 1.815520621,  # 104 01.3
            'ht': 2075.},
        'keck': {  # Keck (Mauna Kea)
            'lat': 0.346040613,  # +19 49.6 (Keck website)
            'lon': 2.71352157,  # 155 28.4
            'ht': 4159.58122},
        'keck2': {
            # http://irtfweb.ifa.hawaii.edu/IRrefdata/telescope_ref_data.html
            'lat': 0.34603876045870413,  # 19 49 35.61788
            'lon': 2.7135372866735916,  # 155 28 27.24268
            'ht': 4159.58122},
        'irtf': {
            # http://irtfweb.ifa.hawaii.edu/IRrefdata/telescope_ref_data.html
            'lat': 0.34603278784504105,  # 19 49 34.38594
            'lon': 2.7134982735227475,  # 155 28 19.19564
            'ht': 4168.06685},
        'clay': {
            'lat': -0.5063938434309143,
            'lon': -1.2338154852026897,
            'ht': 2450.0},
        'clay_jb': {
            'lat': -0.506392081,  # .364" diff ~22 meters different
            'lon': 1.23381854,  # .63" diff
            'ht': 2406.1}
    }
    if type(observatory_loc) != str:
        lat, lon, ht = observatory_loc
    else:
        lat = OBSERVATORY_COORD[observatory_loc]['lat']
        lon = OBSERVATORY_COORD[observatory_loc]['lon']
        ht = OBSERVATORY_COORD[observatory_loc]['ht']

    # Make sure we've got the longitude in the time object to compute lmst
    time = obstime.copy()
    if time.lon == None or type(observatory_loc) != str:
        time.lon = Longitude(lon, unit=u.radian)

    from astropy.utils.iers import IERS_A, IERS_A_URL
    from astropy.utils.data import download_file
    try:
        this_dir, _ = os.path.split(__file__)
        IERS_A_PATH = os.path.join(this_dir, "data", 'finals2000A.all')
        iers_a = IERS_A.open(IERS_A_PATH)
    except IOError:
        iers_a_file = download_file(IERS_A_URL, cache=True)
        iers_a = IERS_A.open(iers_a_file)

    time.delta_ut1_utc = iers_a.ut1_utc(time)

    # Local rotaion rate
    vrot = 465.102 * ((1.0 + 1.57e-7 * ht) /
                      np.sqrt(1.0 + 0.993305 * np.tan(lat) ** 2))

    # Calculate barycentric velocity of earth.
    velh, velb = baryvel(time.jd, 0.0)

    # Find lmst of observation
    lmst = time.sidereal_time('mean')

    # Calculation of geocentric velocity of observatory.
    velt = vrot * np.array([-np.sin(lmst.radian), np.cos(lmst.radian), 0.0])

    # Calculate dv (what is dv?)
    dv = velb + velt * 1e-3

    if sun or heliocentric:
        dv = velh + velt * 1e-3

    if sun:
        import ephem
        eo = ephem.Observer()
        eo.lon = lon
        eo.lat = lat
        eo.elevation = 2450.0
        eo.date = time.datetime
        sunephem = ephem.Sun(eo)

        # Calculation of barycentric velocity components.
        if aORg == 'a':
            sra = np.sin(sunephem.a_ra)
            sdec = np.sin(sunephem.a_dec)
            cra = np.cos(sunephem.a_ra)
            cdec = np.cos(sunephem.a_dec)
        else:
            sra = np.sin(sunephem.g_ra)
            sdec = np.sin(sunephem.g_dec)
            cra = np.cos(sunephem.g_ra)
            cdec = np.cos(sunephem.g_dec)
    else:
        # Make sure coords is an ICRS object
        if type(coords) != ICRS:
            if coords[3] != 2000:
                raise ValueError('Provide IRCS to handle equniox other than 2000')
            coords = ICRS(coords[0], coords[1],
                          unit=(u.degree, u.degree),
                          equinox=Time('J2000', scale='utc'))

        # Precess coordinates to j
        pcoords = coords.fk5.precess_to(time)

        # Calculation of barycentric velocity components.
        sra = np.sin(pcoords.ra.radian)
        sdec = np.sin(pcoords.dec.radian)
        cra = np.cos(pcoords.ra.radian)
        cdec = np.cos(pcoords.dec.radian)

    dvr = dv[0] * cra * cdec + dv[1] * sra * cdec + dv[2] * sdec
    dva = -dv[0] * sra + dv[1] * cra
    dvd = -dv[0] * cra * sdec - dv[1] * sra * sdec + dv[2] * cdec
    dvt = np.sqrt(dva * dva + dvd * dvd)

    # Return
    return [dvr, dvt, dva, dvd]


def photometric_uncertainty(wave, spec, snr=None, mask=None):
    """Return photometric uncertainty in spectrum in m/s."""
    import ipdb;
    ipdb.set_trace()
    if isinstance(snr, type(None)):
        weight = np.ones_like(wave, dtype=np.float)
    elif isinstance(snr, (float, int)):
        weight = np.zeros(len(wave) - 1, dtype=np.float) + snr
    else:
        assert len(snr) == len(wave)
        weight = .5 * (snr[1:] + snr[:-1])

    dellam = np.abs(wave[1:] - wave[:-1])
    dv = SPEED_OF_LIGHT * dellam / wave.mean()
    di = np.abs(spec[1:] - spec[:-1])
    didv = di / dv
    pixel_sigma = didv * weight
    if mask is not None:
        pixel_sigma[mask[1:] | mask[:-1]] = 0.0
    pixel_sigma[~np.isfinite(pixel_sigma)] = 0.0

    return 1.0 / np.sqrt(np.sum(pixel_sigma ** 2.0))


def broaden(wave, spec, dl, usepya=True):
    """ 
    Broaden a spectrum by a gaussian of FWHM dl, uses IUS
    
    Sepectrum should be padded by ~ 5 FWHM on either end
    """

    sig = dl / (2.0 * np.sqrt(2 * np.log(2)))

    #    n=np.ceil((wave.max()-wave.min())/(sig*0.1))
    #    w_lin = np.arange(n, dtype=float)*sig*0.1+wave.min()
    w_lin = np.linspace(wave.min(), wave.max(), wave.size)
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    s_lin = IUS(wave, spec)(w_lin)
    if usepya:
        import PyAstronomy.pyasl
        return w_lin, PyAstronomy.pyasl.broadGaussFast(w_lin, s_lin, sig)

    k_x = np.arange(101, dtype=float) * 0.1 * sig - 5.0 * sig

    kernel = np.exp(-0.5 * (k_x / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))

    import scipy.signal
    broadened = scipy.signal.fftconvolve(s_lin, kernel, 'same') * 0.1 * sig

    return w_lin, broadened


def avgstd(values, weights=None, ret_e=False,
           bootstrapeN=1000, ret_std_e=False):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    
    axis must be first axis if specified
    
    ret_e assumes weights are 1/var
    see also http://stats.stackexchange.com/questions/25895/computing-standard-error-in-weighted-mean-estimation
    """
    if not isinstance(values, np.ndarray): values = np.array(values)
    if not isinstance(weights, np.ndarray): weights = np.array(weights)

    if values.ndim != weights.ndim:
        raise ValueError('Array dimensions do not match')

    values = np.squeeze(values)
    weights = np.squeeze(weights)

    axis = None if values.ndim == 1 else 0

    if axis is not None and (ret_e or ret_std_e):
        import ipdb;
        ipdb.set_trace()
        raise ValueError('Axis not supported')

    try:
        average = np.average(values, weights=weights, axis=axis)
        variance = (((values - average) ** 2 * weights).sum(axis) /
                    weights.sum(axis) - (weights ** 2).sum(axis) / weights.sum(axis))
        ret = [average, np.sqrt(variance)]
        if ret_e or ret_std_e:
            bootstrap_e = bootstrap_weightedmean_err(values, 1 / np.sqrt(weights), N=bootstrapeN)
            ret.append(bootstrap_e[0])
            if ret_std_e:
                ret.append(bootstrap_e[1])

        return tuple(ret)
    except ZeroDivisionError:
        return (np.nan, np.nan, np.nan) if ret_e else (np.nan, np.nan)


def bootstrap_weightedmean_err(xi, si, N=1000):
    """" sample the gaussians at (xi,si) N times"""
    ind = np.arange(xi.size)
    bp = np.array([avgstd(*np.array([(np.random.normal(xi[i], si[i]), 1 / si[i] ** 2)
                                     for i in
                                     np.random.choice(ind, ind.size, replace=True)]
                                    ).T, ret_e=False)[:2] for n in range(N)])
    return bp.std(0)


def avgerr(values, weights, axis=None):
    """
    Return the weighted average and its error.
    
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=axis)
    variance = 1 / (weights.sum(axis))
    return (average, np.sqrt(variance))


def color_z_plot(x, y, z, cmap_name='nipy_spectral', lim=1e100, psym='o',
                 label=None, cbax=None, nocbar=False):
    import matplotlib.pyplot as plt
    oset = z - np.median(z)
    good = (np.abs(oset) < lim)
    cmap = plt.cm.get_cmap(cmap_name)
    c = oset[good] - oset[good].min()
    c = (255 * c / c.max()).round().astype(int).clip(0, 255)

    for i, ci in enumerate(c): plt.plot(x[good][i], y[good][i], 'o', c=cmap(ci))
    plt.plot(x[oset < -lim], y[oset < -lim], psym, c=cmap(0))
    plt.plot(x[oset > lim], y[oset > lim], psym, c=cmap(255))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=z[good].min(), vmax=z[good].max()))
    sm._A = []
    if not nocbar:
        cbar = plt.colorbar(sm, cax=cbax)
        if label is not None: cbar.set_label(label)
        return sm, cbar
    else:
        return sm, None

def binned_xy_plot(x, y, nbin=10):
    import matplotlib.pyplot as plt
    x = np.array(x)
    y = np.array(y)

    use = np.isfinite(x) & np.isfinite(y)
    x = x[use]
    y = y[use]

    n, _ = np.histogram(x, bins=nbin)
    sy, _ = np.histogram(x, bins=nbin, weights=y)
    sy2, _ = np.histogram(x, bins=nbin, weights=y * y)
    mean = sy / n
    std = np.sqrt(sy2 / n - mean ** 2)

    plt.plot(x, y, 'bo')
    plt.errorbar((_[1:] + _[:-1]) / 2, mean, yerr=std, fmt='r-')


def casagrandeTeff(color, sys='BV', feh=0.0, clip=False, colore=0):
    data = {'BV': {'mr': (-5.0, 0.4), 'xr': (0.18, 1.29), 'e': 82,
                   'ai': (0.5665, 0.4809, -0.0060, -0.0613, -0.0042, 0.0035)},
            'VRc': {'mr': (-5.0, 0.3), 'xr': (0.24, 0.80), 'e': 59,
                    'ai': (0.4386, 1.4614, -0.7014, -0.0807, 0.0142, 0.0012)},
            'RcIc': {'mr': (-5.0, 0.3), 'xr': (0.23, 0.68), 'e': 42,
                     'ai': (0.3296, 1.9716, -1.0225, -0.0298, 0.0329, 0.0011)},
            'VIc': {'mr': (-5.0, 0.3), 'xr': (0.46, 1.47), 'e': 33,
                    'ai': (0.4033, 0.8171, -0.1987, -0.0409, 0.0319, 0.0025)},
            'VJ': {'mr': (-5.0, 0.4), 'xr': (0.61, 2.44), 'e': 25,
                   'ai': (0.4669, 0.3849, -0.0350, -0.0140, 0.0225, 0.0016)},
            'VH': {'mr': (-5.0, 0.4), 'xr': (0.67, 3.01), 'e': 132,
                   'ai': (0.5251, 0.2553, -0.0119, -0.0187, 0.0410, 0.002)},
            'VKs': {'mr': (-5.0, 0.4), 'xr': (0.78, 3.15), 'e': 79,
                    'ai': (0.5057, 0.2600, -0.0146, -0.0131, 0.0288, -0.0087)},
            'JKs': {'mr': (-5.0, 0.4), 'xr': (0.07, 0.80), 'e': 43,
                    'ai': (0.6393, 0.6104, 0.0920, -0.0330, 0.0291, -0.0009)},
            'BtVt': {'mr': (-2.7, 0.4), 'xr': (0.19, 1.49), 'e': 26,
                     'ai': (0.5839, 0.4000, -0.0067, -0.0282, -0.0346, 0.0021)},
            'VtJ': {'mr': (-2.7, 0.4), 'xr': (0.77, 2.56), 'e': 18,
                    'ai': (0.4525, 0.3797, -0.0357, -0.0082, 0.0123, -0.0001)},
            'VtH': {'mr': (-2.7, 0.4), 'xr': (0.77, 3.16), 'e': 62,
                    'ai': (0.5286, 0.2354, -0.0073, -0.0182, 0.0401, -0.0055)},
            'VtKs': {'mr': (-2.4, 0.4), 'xr': (0.99, 3.29), 'e': 73,
                     'ai': (0.4892, 0.2634, -0.0165, -0.0121, 0.0249, -0.0055)},
            'by': {'mr': (-3.7, 0.5), 'xr': (0.18, 0.72), 'e': 62,
                   'ai': (0.5796, 0.4812, 0.5747, -0.0633, 0.0042, -0.0015)}}
    if sys not in data:
        raise ValueError('Valid systems are {}'.format(', '.join(data.keys())))
    dat = data[sys]
    if feh < dat['mr'][0] or feh > dat['mr'][1]:
        if clip:
            print('Clipping [Fe/H]')
            feh = min(max(feh, dat['mr'][0]), dat['mr'][1])
        else:
            raise ValueError('Valid [Fe/H] range is {}'.format(dat['mr']))
    if color < dat['xr'][0] or color > dat['xr'][1]:
        if clip:
            print('Clipping color')
            color = min(max(color, dat['xr'][0]), dat['xr'][1])
        else:
            raise ValueError('Valid color range is {}'.format(dat['xr']))
    a0, a1, a2, a3, a4, a5 = dat['ai']

    dtdc = (- 5040.0 * (a1 + a3 * feh + 2 * a2 * color) /
            (a0 + a4 * feh + a5 * feh ** 2 + a1 * color + a3 * feh * color + a2 * color ** 2) ** 2)
    dtdc * colore

    return (5040.0 / (a0 + a1 * color + a2 * color ** 2 + a3 * color * feh +
                      a4 * feh + a5 * feh ** 2),
            np.sqrt((dat['e'] + 17) ** 2 + (dtdc * colore) ** 2))


# def plotSpectrum(y,Fs):
#    """
#        Plots a Single-Sided Amplitude Spectrum of y(t)
#        """
#    n = len(y) # length of the signal
#    k = arange(n)
#    T = n/Fs
#    frq = k/T # two sides frequency range
#    frq = frq[range(n/2)] # one side frequency range
#    
#    Y = fft.fft(y)/n # fft computing and normalization
#    Y = Y[range(n/2)]
#    
#    plot(frq,abs(Y)) # plotting the spectrum
#    xlabel('Freq (Hz)')
#    ylabel('|Y(freq)|')
#
# Fs = 150.0;  # sampling rate
# Ts = 1.0/Fs; # sampling interval
# t = arange(0,1,Ts) # time vector
#
# ff = 5;   # frequency of the signal
# y = sin(2*pi*ff*t)
#
# subplot(2,1,1)
# plot(t,y)
# xlabel('Time')
# ylabel('Amplitude')
# subplot(2,1,2)
# plotSpectrum(y,Fs)
# show()

def sigma_clip_polyfit(x, y, power, sig=3, sigu=None, sigl=None, iter=1):
    if sigl == None:
        sigl = sig
    if sigu == None:
        sigu = sig

    good = np.ones_like(x, dtype=bool)
    iter_left = iter
    while iter_left > 0:
        cc = np.polyfit(x[good], y[good], power)
        deviation = y - np.poly1d(cc)(x)
        stdev = np.sqrt((deviation ** 2).sum())
        good = (deviation >= -sigl * stdev) & (deviation <= sigu * stdev)
        iter_left -= 1
    #        figure(3)
    #        plot(x,y,'r.')
    #        plot(x[good],y[good],'*')
    #        figure(4)
    #        plot(x,deviation,'o')
    #        axhline(-sigl * stdev)
    #        axhline( sigu * stdev)
    #        raw_input('?')

    ret = np.poly1d(cc)

    #    from astropy.stats import sigma_clip
    #    xx=np.arange(x.shape[0])
    #    cenfunc = lambda yi: poly1d(np.ma.polyfit(xx, yi, power))(xx)
    #    clipped=sigma_clip(y, sig=sig, iters=iter, cenfunc=cenfunc,
    #                       axis=None, copy=True)
    #    sc_ret=np.poly1d(np.polyfit(x, clipped, power))

    return ret


# clf();mpoly,scpoly=sigma_clip_polyfit(xx,yy,1,sig=1, sigu=.2,iter=10);plot(sorted(xx),mpoly(sorted(xx)),'b');plot(xx,yy,'r.');ylim(0,10000)
#
# ;plot(sorted(xx),scpoly(sorted(xx)),'--g')


def rebin_spec(spec, w_in_center, w_out_center):
    """waves are at the bin centers"""
    # move from centers to leading edge by linear approx
    dw_in_center = np.diff(w_in_center)
    w_in = w_in_center - np.concatenate([dw_in_center[0:1], dw_in_center]) / 2

    dw_out_center = np.diff(w_out_center)
    w_out = w_out_center - np.concatenate([dw_out_center[0:1], dw_out_center])

    # width of input bins by linear approx
    dw_in = np.diff(w_in)
    # extrapolate to get width of last
    dw_in = np.concatenate([dw_in, dw_in[-1:]])

    #    dw_out=np.diff(w_out)
    #    dw_out=np.concatenate([dw_out,dw_out[-1:]])

    # output
    sout = np.zeros_like(w_out, dtype=spec.dtype)
    #    vout=sout.copy()

    #    import ipdb;ipdb.set_trace()
    # Regrid
    for i in range(w_out.size - 1):
        # done, no more data, barring the last fractional pixel
        if w_out[i] > w_in[-1]:
            print('Breaking on {}'.format(i))
            break
        if w_out[i] < w_in[0]: continue  # not fully in region yet

        # find last w_in[j] <= w_out[i]
        j = np.where(w_in <= w_out[i])[0][-1]
        #        if i==1200:
        #            import ipdb;ipdb.set_trace()
        # find last w_in[j] <= w_out[i+1]
        try:
            jl = np.where(w_in[j:] <= w_out[i + 1])[0][-1] + j
        except IndexError:
            break

        # w_in[j:jl+1] are all in the output pixel, at least partially

        if jl == j:  # just taking this one subpixel (new pix is fully inside j)
            #            print 'Multiple in 1'
            frac = (w_out[i + 1] - w_out[i]) / dw_in[j]
            sout[i] = frac * spec[j]
        #            vout[i]=frac**2*var[j]
        else:
            # will take some of this pixel and some or all of j+1,...

            # this pixel
            frac = (w_in[j + 1] - w_out[i]) / dw_in[j]
            sout[i] = frac * spec[j]
            #            vout[i]=frac**2*var[j]

            # we get all of j+1 to jl excluding jl
            sout[i] += spec[j + 1:jl].sum()
            #            vout[i]+=var[j+1:jl].sum()

            # we get some of jl
            frac = (w_out[i + 1] - w_in[jl]) / dw_in[jl]
            sout[i] += frac * spec[jl]
    #            vout[i]+=frac**2*var[jl]
    #
    #        if sout[i]==0:
    #            import ipdb;ipdb.set_trace()

    return sout  # , vout


def massradius_torres(teff, logg, feh):
    """
      Uses the Torres relation
      (http://adsabs.harvard.edu/abs/2010A%26ARv..18...67T) to determine
      the stellar mass and radius given the logg, Teff, and [Fe/H].

      NOTE the errors in the empirical model of ulogm = 0.027d0, ulogr = 0.014d0

    INPUTS:
       LOGG - The log of the stellar surface gravity
       TEFF - The stellar effective temperature
       FEH  - The stellar metalicity

    OUTPUTS:
       MSTAR - The stellar mass, in solar masses
       RSTAR - The stellar radius, in sexitolar radii

    MODIFICATION HISTORY

     2012/06 -- Public release -- Jason Eastman (LCOGT)
     2015/02 -- Convert to python -- Jeb Bailey (UMichigan)
    """
    if type(teff) in (int, float): teff = np.array([teff])
    if type(logg) in (int, float): logg = np.array([logg])
    if type(feh) in (int, float): feh = np.array([feh])

    teff = np.asarray(teff)
    logg = np.asarray(logg)
    feh = np.asarray(feh)

    # coefficients from Torres, 2010
    ai = np.array([1.5689, 1.3787, 0.4243, 1.139, -0.14250, 0.01969, 0.10100])
    bi = np.array([2.4427, 0.6679, 0.1771, 0.705, -0.21415, 0.02306, 0.04173])

    # ulogm = 0.027d0
    # ulogr = 0.014d0

    X = np.log10(teff) - 4.1

    coeffs = np.array([1.0, X, X ** 2.0, X ** 3.0, logg ** 2.0, logg ** 3.0, feh])
    logm = (ai * coeffs).sum(0)
    logr = (bi * coeffs).sum(0)

    return 10.0 ** logm, 10.0 ** logr


def est_companion_mass(M, P, sRV):
    """
    mass of star in Msun, P in days, RV induced is peak to peak
    """
    twopi = 2.0 * np.pi
    Msun = 1.989e33  # g
    G = 6.674e-8  # cgs: cm^3 / g s^2
    # AU = 1.496e13  # cm
    secpday = 24.0 * 3600  # s/day
    # secpyear = 24.0 * 3600 * 365.25  # s/year
    a = ((G * M * Msun * (P * secpday) ** 2 / twopi ** 2) ** (1 / 3.0))
    return M * 1047.889 * (sRV * 1e2) / np.sqrt(G * M * Msun / a) / 2
