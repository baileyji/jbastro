import numpy as np
import matplotlib.pyplot as plt
import astropy
import math
from great_circle_dist import dist_radec_fast

M2FS_FOV_DEG=29.0/60

from astropy.io import fits
from . lacosmics.cosmics import cosmicsimage

def crreject(im, **cosmics_settings):
    """Give a seqno or a path to a quad if file set"""
    def_cosmic_settings={'sigclip': 6.0, 'sigfrac': 0.5,
        'objlim': 1.4, 'iter':7,'readnoise':0.0,
        'gain':1.0, 'satlevel':.95*(2**16)}
    
    for k, v in def_cosmic_settings.iteritems():
        if k not in cosmics_settings:
            cosmics_settings[k]=v
    
    cosmic_iter=cosmics_settings.pop('iter')
    
    c=cosmicsimage(im, **cosmics_settings)
    c.run(maxiter = cosmic_iter)
    
    return c.mask.astype(np.uint8)

def sex2dec(d,m,s,ra=False):
    """ 
    Convert string/int/float args to decimal degrees
    remember -0 doesn't exist in python as an int
    """
    mul=15.0 if ra else 1.0
    sign=math.copysign(1.0,float(d))
    if sign<0:
        return mul*(float(d) - float(m)/60.0 - float(s)/3600.0)
    else:
        return mul*(float(d) + float(m)/60.0 + float(s)/3600.0)

def dec2sex(n, ra=False):
    """ Convert decimal degrees to h, m, s or d, m ,s """
    if type(n) not in (float, int):
        raise ValueError('Must give d as float or int')
    
    sign=-1.0 if n < 0 else 1.0
    n=abs(float(n))
    if ra:
        n/=15.0
        hord=int(n)
        m=int((n-hord)*60)
        secs=(n-hord)*3600-m*60
    
    else:
        hord=int(n)
        m=int((n-hord)*60)
        secs=(n-hord)*3600-m*60
    
    return sign*hord,m,secs,

#def sexconvert(*args,**kwargs):
#    """Hack to force Matt's rounding errors"""
#    ra=kwargs.get('ra',False)
#    dtype=kwargs.get('dtype',str)
#    fmt=kwargs.get('fmt','{: 03.0f}:{:02}:{:07.4f}')
#    if len(args)==1:
#        x=args[0]
#    elif len(args)==3:
#        x=args[:3]
#    else:
#        raise ValueError('Unsupported args: {}',str(args))
#    
#    x=_sexconvert(x,dtype=float,ra=ra)
#    myfmt='{: 03.0f}:{:02}:{:05.2f}' if ra else '{: 03.0f}:{:02}:{:04.1f}'
#    x=_sexconvert(x,dtype=str,ra=ra,fmt=myfmt)
#    return _sexconvert(x,dtype=dtype,ra=ra,fmt=fmt)

def sexconvert(*args,**kwargs):
    """convert a sexgesmal number to something """
    ra=kwargs.get('ra',False)
    dtype=kwargs.get('dtype',str)
    fmt=kwargs.get('fmt','{: 03.0f}:{:02}:{:07.4f}')
    if len(args)==1:
        x=args[0]
    elif len(args)==3:
        x=args[:3]
    else:
        raise ValueError('Unsupported args: {}',str(args))

    if dtype not in [str,float]:
         raise ValueError('type {} unsupported',str(dtype))
    
    try:
        x=float(x)
    except (TypeError, ValueError):
        pass

    if type(x) == float:
        if dtype==float:
            return float(x)
        else:
            return fmt.format(*dec2sex(x,ra=ra))
    elif type(x)==str:
        x=x.strip()
        x=x.split(':')
        if len(x)==1:
            x=x[0].split()
        x=sex2dec(*x,ra=ra)
        if dtype==float:
            return x
        else:
            return fmt.format(*dec2sex(x,ra=ra))
    elif len(x)==3:
        try:
            x=sex2dec(*x,ra=ra)
        except Exception:
            raise ValueError('Unsupported args: {}',str(args))
        if dtype==float:
            return x
        else:
            return fmt.format(*dec2sex(x,ra=ra))
    else:
        raise ValueError('Unsupported args: {}',str(args))

def test_sexconvert():

    def test_inner(ra_or_dec, strs, floats):
        try:
            #str->str
            for i,v in enumerate(strs):
                assert sexconvert(v, dtype=str, ra=ra_or_dec).strip()==v+'00'
            #str->float
            for i,v in enumerate(strs):
                assert sexconvert(v, dtype=float, ra=ra_or_dec)==floats[i]
            #float->float
            for i,v in enumerate(floats):
                assert sexconvert(v, dtype=float, ra=ra_or_dec)==v
            #float->str
            for i,v in enumerate(floats):
                assert sexconvert(v, dtype=str, ra=ra_or_dec).strip()==strs[i]+'00'
        except AssertionError,e:
            print str(e)
            import ipdb;ipdb.set_trace()
    test_inner(True,
               ['11:12:09.85','-00:34:32.02','00:34:32.02','-10:34:32.02'],
               [15*(11+12.0/60+9.85/3600), -15*(0+34.0/60+32.02/3600),
                15*(0+34.0/60+32.02/3600), -15*(10+34.0/60+32.02/3600)])

    test_inner(False,
               ['71:12:09.85','-00:34:32.02','00:34:32.02','-80:34:32.02'],
               [(71+12.0/60+9.85/3600), -(0+34.0/60+32.02/3600),
                (0+34.0/60+32.02/3600), -(80+34.0/60+32.02/3600)])

test_sexconvert()


def cycscatter(*args,**kwargs):
    """Make a cyclic scatter plot"""
    x=args[0].copy()
    if 'set_xlim' in kwargs:
        set_xlim=kwargs.pop('set_xlim')
    else:
        set_xlim=False
    if 'forcecyc' in kwargs:
        force=kwargs.pop('forcecyc')
    else:
        force=False
    if force or (len(np.where(x<100)[0])>0 and len(np.where(x>300)[0])>0):
        if x.shape==():
            x=x+360.
        else:
            x[x<100]=x[x<100]+360.
        from matplotlib.ticker import FuncFormatter
        @FuncFormatter
        def wrapticks(x,pos):
            return "{:.1f}".format(x % 360)
        #tmp=plt.gca().xaxis.get_major_formatter()
        plt.gca().xaxis.set_major_formatter(wrapticks)
        plt.scatter(x,*args[1:],**kwargs)
        if set_xlim:
            plt.gca().set_xlim((min(x),max(x)))
        #plt.gca().xaxis.set_major_formatter(tmp)
        return True
    else:
        plt.scatter(*args,**kwargs)
        return False

def getIsochrone(age, color='VJ'):
    """Takes age in Myr color=vj or bv"""
    if age < 100:
        grid_age=roundTo(age, 10)
    elif age < 500:
        grid_age=roundTo(age, 50)
    elif age < 1000:
        grid_age=roundTo(age, 100)
    elif age < 3500:
        grid_age=roundTo(age, 250)
    elif age < 3700:
        grid_age=roundTo(age, 100)
    elif age < 3800:
        grid_age=roundTo(age, 50)
    elif age < 4000:
        grid_age=roundTo(age, 100)
    elif age < 16250:
        grid_age=roundTo(age, 500)
    else:
        raise ValueError("No Isochrone for {0}".format(age))
    isofile='isoz22s_c03hbs/wzsunysuns.t6{grid_age:05}_c03hbs'
    
    #Load V & V-J (col 9) NB B-v is col 6
    if color.lower()=='vj':
        return np.loadtxt(isofile.format(grid_age=grid_age), comments='#',
                          unpack=True, usecols=(4,9))
    elif color.lower()=='bv':
        return np.loadtxt(isofile.format(grid_age=grid_age), comments='#',
                          unpack=True, usecols=(4,6))


def RtoV(Rabs):
    """
    Rabs -> Vabs F5-K5
    Interpolates based on values for F5,G5, K3, & K5 MS dwarfs
    Dom=[3.17,6.53]
    """ 
    ygrid=np.array([3.5,5.1,6.8,7.5])
    xgrid=np.array([3.5-.33,5.1-.47,6.8-.8,7.5-.97])
    return np.interp(Rabs,xgrid,ygrid)


def roundTo(x, value):
    """ Round to the nearest value """
    return int(round(x/value))*value


#def expTime(paramTriple):
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


def expTime(paramTriple, seeing=1.0):
    """ (SNR, mag, expTime) """
    snr, mag, t = paramTriple

    zp= .2 * .5 # R factor * e/s guess per Mario
    zpm=17.5
    zpm-=(seeing-1.0)*2
    
    #TODO proper seeing correction
    # integrate moffat with seeing aperture and compare to arerture size
    if snr is None:
        snr = np.sqrt(zp * 10**((mag-zpm)/-2.5) * 3600 * t )
    if mag is None:
        if snr <=0:
            mag=float('inf')
        else:
            mag=np.log10((snr**2)/zp/3600/t)*-2.5 +zpm
    if t is None:
        t=(snr**2)/(zp * 10**((mag-zpm)/-2.5) * 3600)
    return (snr, mag, t)


def estMag(sptype, band='R'):
    """V or R (default) abs mag for a B0-M0 star """
    try:
        mult=['B','A','F','G','K', 'M'].index(sptype[0].upper())
    except ValueError:
        raise ValueError('Invalid spectral type ({})'.format(sptype))

    x=np.array([0, 1, 2, 3, 4,
                5, 8, 10, 12, 15,
                17, 20, 22, 25, 28,
                30, 32, 35, 38, 40,
                42, 43, 45, 47, 50])
    VmR=np.array([-.13, -.11, -.1, -.08, -.07,
                  -.06, -.02, 0.02, 0.08, 0.16,
                  0.19, 0.3, 0.35, 0.4, 0.47,
                  0.5, 0.53, 0.54, 0.58, 0.64,
                  0.74, 0.81, 0.99, 1.15, 1.28])
    V=np.array([-3.3, -2.9, -2.5, -2.0, -1.5,
                -1.1, 0.0, 0.7, 1.3, 1.9,
                2.3, 2.7, 3.0, 3.5, 4.0,
                4.4, 4.7, 5.1, 5.6, 6.0,
                6.5, 6.8, 7.5, 8.0, 8.8])
    if band == 'R':
        return np.interp(int(sptype[1])+mult*10, x, V-VmR)
    elif band =='V':
        return np.interp(int(sptype[1])+mult*10, x, V)

def estBV(sptype):
    """ 
    B-V value for a B0-M0 star
    
    Appendix B, Gray
    """
    try:
        mult=['B','A','F','G','K', 'M'].index(sptype[0].upper())
    except ValueError:
        raise ValueError('Invalid spectral type ({})'.format(sptype))
    return np.interp(int(sptype[1])+mult*10,
                     np.array([0, 1, 2, 3, 4,
                               5, 8, 10, 12, 15,
                               17, 20, 22, 25, 28,
                               30, 32, 35, 38, 40,
                               42, 43, 45, 47, 50]),
                     np.array([-.29, -.26, -.24, -.21, -.18,
                               -.16, -.10,  0.0, 0.06, 0.14,
                               0.19, 0.31, 0.36, 0.44, 0.53,
                               0.59, 0.63, 0.68, 0.74, 0.82,
                               0.92, 0.96, 1.15, 1.30, 1.41]))

def estSpType(absmag, dm=None, band='V'):
    """Spectral type for Mag, mag if dm=modulus, V(default) or R"""
    if dm != None:
        absmag=absmag-dm
    if band=='V':
        type=int(round(np.interp(absmag, np.array([3.5,5.1,6.8,7.5]),
                                 np.array([0,10,18,20]))))
    elif band =='R':
        type=int(round(np.interp(absmag,
                                 np.array([3.5-.33,5.1-.47,6.8-.8,7.5-.97]),
                                 np.array([0,10,18,20]))))
    if type<5:
        type='F'+str((type % 10)+5)[-1]
    elif type < 15:
        type='G'+str((type % 10)+5)[-1]
    elif type <21:
        type='K'+str((type % 10)+5)[-1]
    return type

def massLuminosity(mass):
    if mass<.43:
        k=.23
        a=2.3
    elif mass<2:
        k=1.0
        a=4
    elif mass < 20:
        k=1.5
        a=3.5
    else:
        raise ValueError("Mass too large")
    if mass < 0.08:
        alpha=.3
    if mass < 0.5:
        alpha=1.3
    else:
        alpha=2.3
    beta=alpha/a/2.5
    omega=k**(alpha/a)
    j=(10.0**(beta*c1)-10.0**(beta*c2))
    mbol=(-1.0/b)*np.log10(N*b*np.log(10)/o/j)

def VtoR(Vabs):
    xgrid=np.array([3.5,5.1,6.8,7.5])
    ygrid=np.array([3.5-.33,5.1-.47,6.8-.8,7.5-.97])
    return np.interp(Vabs,xgrid,ygrid)

def rvPrecision(snr, sptype='K5'):
    xgrid=np.array([15,35,55,75,95,115,175,235,295,315,375])
    ygrid=np.array([236,101,64,47,37,31,20,15,12,11,9])
    sigma=int(round(np.interp(snr,xgrid,ygrid)))
    if 'G' in sptype:
        sigma*=1.065 #emperic estimate from IDL numbers
    elif 'F' in sptype:
        sigma*=1.59
    return sigma
#
#S/N      Atm          F5          K5          G5
#15       179         375         236         251
#35        77         161         101         108
#55        49         102          64          68
#75        36          75          47          50
#95        28          59          37          40
#115       23          49          31          33
#175       15          32          20          22
#235       11          24          15          16
#295        9          19          12          13
#315        9          18          11          12
#375        7          15           9          10

def obsTimes(snr, dm):
    #F5
    Rmag=dm+3.5-.33
    f5t=expTime((snr,Rmag,None))[2]
    #G5
    Rmag=dm+5.1-.47
    g5t=expTime((snr,Rmag,None))[2]
    #K3
    Rmag=dm+6.8-.8
    k3t=expTime((snr,Rmag,None))[2]
    #K5
    Rmag=dm+7.5-.97
    k5t=expTime((snr,Rmag,None))[2]
    #3hr depth
    mr=expTime((snr,None,3))[1]-dm
    print ("Time to {0} for F5:{1:.2f} G5:{2:.2f}"+
          "K3:{k3:.2f} K5:{3:.2f}. 3h to MR={4:.1f}").format(
                                    snr, f5t,g5t,k5t,mr, k3=k3t)



def where2bool(length, whereTrue):
    out=np.zeros(length,dtype=np.bool)
    out[whereTrue]=True
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
        ctr=(coord['RAJ2000'],coord['DEJ2000'])
    elif type(coord) ==astropy.coordinates.builtin_systems.ICRSCoordinates:
        ctr=(coord.ra.degrees,coord.dec.degrees)
    else:
        ctr=coord
    
    if type(stars) == astropy.io.fits.fitsrec.FITS_rec:
        ras=stars['RAJ2000']
        des=stars['DEJ2000']
    else:
        ras=stars[0]
        des=stars[1]
    try:
        #Extract a square of stars surrounding the field
        ra_hwid=fov/np.cos(ctr[1]*np.pi/180.0)/2.0
        de_hwid=fov/2.0
        
        ra_min=ctr[0]-ra_hwid
        ra_max=ctr[0]+ra_hwid
        de_min=ctr[1]-de_hwid
        de_max=ctr[1]+de_hwid
        if ra_min > ra_max:
            ra_cut=((ras > ra_min) |
                    (ras < ra_max))
        else:
            ra_cut=((ras > ra_min) &
                    (ras < ra_max))
        
        de_cut=((des > de_min) & (des < de_max))
        
        cand=ra_cut & de_cut
        
        if not square:
            #Now make sure they are all within the field
            sep=dist_radec_fast(ctr[0], ctr[1],
                                ras[cand], des[cand],
                                method='Haversine', unit='deg',
                                scale=fov/2.0)
            cand[cand]=sep < (fov/2.0)
    except Exception:
        import ipdb;ipdb.set_trace()
    if mask:
        return cand
    else:
        return np.where(cand)[0]

def gaussfit(xdata, ydata):
    def gauss_quad(x, a0, a1, a2, a3, a4, a5):
        z = (x - a1) / a2
        y = a0 * np.exp(-z**2 / a2) + a3 + a4 * x + a5 * x**2
        return y

    from scipy.optimize import curve_fit
    parameters, covariance = curve_fit(gauss_quad, xdata, ydata)

    return (parameters, gauss_quad(xdata, *parameters))



def aniscatter(x,y, **kwargs):
    import matplotlib.animation as animation
    numframes = len(x)
    
    try:
        interval=kwargs.pop('interval')
    except Exception:
        interval=200

    if 'marker' not in kwargs:
        kwargs['marker']='o'
    
    fig = plt.gcf()
    line, = plt.plot(x[[]], y[[]], linestyle='none', **kwargs)
    
    def update_plot(i):
        line.set_data(x[:i],y[:i])
        return line,
    
    ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),
                                  interval=interval, repeat=False)
    plt.show()
    #display_animation(ani)

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
    plt.close(anim._fig)
    from IPython.display import HTML
    return HTML(anim_to_html(anim))


def extract_key_from_FITSrec_list(targetThing, key):
    """
    Return a list of two arrays, RAs & DECs. Creates copy of coords.
    """
    if type(targetThing) == astropy.io.fits.fitsrec.FITS_rec:
        out=targetThing[key].copy()
    else:
        t_lens=map(len, targetThing)
        n=sum(t_lens)
        out=np.zeros(n, dtype=targetThing[0][key].dtype)
        for i,arr in enumerate(targetThing):
            ndx=sum(t_lens[0:i])
            out[ndx:ndx+len(arr)]=arr[key]
    return out

def pltradec(thing,clear=False,lw=0,m='.',c='k',fig=None):
    """thing must have keys RAJ2000 and DEJ2000 """
    #global clusters
    if fig!=None:
        plt.figure(fig)
    if clear:
        plt.clf()
    plt.scatter(thing['RAJ2000'],thing['DEJ2000'],c=c,marker=m,lw=lw)
    plt.ylabel('Dec')
    plt.xlabel('RA')
    plt.show()

def dm2d(dm):
    return int(round(10.0**((dm+5.0)/5.0)))

def d2dm(parsec):
    return -5 + 5.0*np.log10(parsec)


def baryvel_los(obstime, coords, observatory_loc):
    """
    vvec (output vector(4))
    Various projections of the barycentric velocity
	correction, expressed in km/sec. The four elements in the vector are radial,
 	tangential, right ascension, and declination projections respectively.
 	Add vvec(0) to the observed velocity scale to shift it to the barycenter.
    """
    from PyAstronomy.pyasl import baryvel
    from astropy.coordinates import Longitude, ICRS
    import astropy.units as u
    import os
    
    OBSERVATORY_COORD={
     'lick3':{
        'lat':0.651734547,   #+37 20 29.9
        'lon':2.123019229,	  # 121 38 24.15
        'ht':1283.},
     'cfht': {
        'lat':0.346030917,  #+19 49 34
        'lon':2.713492477, #155 28 18
        'ht':4198.},
     'kp':{
        'lat':0.557865407,	#+31 57.8 (1991 Almanac)
        'lon': 1.947787445,  #111 36.0
        'ht':2120.},
     'mcd':{ #McDonald (Fort Davis)
	    'lat':0.5353215959, # +30 40.3 (1995 Almanac)
	    'lon':1.815520621, #104 01.3
        'ht':2075.},
     'keck':{ #Keck (Mauna Kea)
	    'lat':0.346040613,  #+19 49.6 (Keck website)
	    'lon': 2.71352157,  #155 28.4
        'ht':4159.58122},
     'keck2':{
         #http://irtfweb.ifa.hawaii.edu/IRrefdata/telescope_ref_data.html
         'lat':0.34603876045870413, #19 49 35.61788
        'lon':2.7135372866735916,   #155 28 27.24268
        'ht':4159.58122},
     'irtf':{
         #http://irtfweb.ifa.hawaii.edu/IRrefdata/telescope_ref_data.html
	    'lat':0.34603278784504105, #19 49 34.38594
	    'lon':2.7134982735227475,  #155 28 19.19564
        'ht':4168.06685},
     'clay':{
    	  'lat':-0.5063938434309143,
    	  'lon':-1.2338154852026897,
    	  'ht':2450.0}
    }
    if type(observatory_loc)!=str:
        lat, lon, ht =observatory_loc
    else:
        lat=OBSERVATORY_COORD[observatory_loc]['lat']
        lon=OBSERVATORY_COORD[observatory_loc]['lon']
        ht=OBSERVATORY_COORD[observatory_loc]['ht']

    #Make sure we've got the longitude in the time object to compute lmst
    time=obstime.copy()
    if time.lon==None:
        time.lon=Longitude(lon, unit=u.radian)

    from astropy.utils.iers import IERS_A,IERS_A_URL
    from astropy.utils.data import download_file
    try:
        this_dir, _ = os.path.split(__file__)
        IERS_A_PATH = os.path.join(this_dir, "data", 'finals2000A.all')
        iers_a = IERS_A.open(IERS_A_PATH)
    except IOError:
        iers_a_file = download_file(IERS_A_URL, cache=True)
        iers_a = IERS_A.open(iers_a_file)

    time.delta_ut1_utc = iers_a.ut1_utc(time)


    #Make sure coords is an ICRS object
    if type(coords) != ICRS:
        if coords[3]!=2000:
            raise ValueError('Provide IRCS to handle equniox other than 2000')
        coords=ICRS(coords[0], coords[1],
                    unit=(u.degree, u.degree),
                    equinox=Time('J2000', scale='utc'))

    #Local rotaion rate
    vrot = 465.102 * ((1.0 + 1.57e-7 * ht) /
                     np.sqrt(1.0 + 0.993305 * np.tan(lat)**2))

    #Precess coordinates to j
    pcoords=coords.fk5.precess_to(time)

    #Calculate barycentric velocity of earth.
    velh, velb= baryvel(time.jd,0.0)

    #Find lmst of observation
    lmst=time.sidereal_time('mean')

    #Calculation of geocentric velocity of observatory.
    velt = vrot * np.array([-np.sin(lmst.radian), np.cos(lmst.radian), 0.0])

    #Calculation of barycentric velocity components.
    sra = np.sin(pcoords.ra.radian)
    sdec = np.sin(pcoords.dec.radian)
    cra = np.cos(pcoords.ra.radian)
    cdec = np.cos(pcoords.dec.radian)

    dv = velb + velt * 1e-3
    dvr = dv[0]*cra*cdec + dv[1]*sra*cdec + dv[2]*sdec
    dva = -dv[0]*sra + dv[1]*cra
    dvd = -dv[0]*cra*sdec - dv[1]*sra*sdec + dv[2]*cdec
    dvt = np.sqrt(dva*dva + dvd*dvd)

    #Return
    return [dvr, dvt, dva, dvd]
