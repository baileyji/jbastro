import numpy as np
import matplotlib.pyplot as plt
import astropy
from great_circle_dist import dist_radec_fast

M2FS_FOV_DEG=29.0/60

def cycscatter(*args,**kwargs):
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
    """Takes age in Myr"""
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
    """Get the Vabs for a star with Rabs
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
    """ Return the V or R abs mag for a B0-M0 star """
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
    Return the B-V value for a B0-M0 star
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
    if dm != None:
        absmag=absmag-dm
    if band=='V':
        type=int(round(np.interp(absmag, np.array([3.5,5.1,6.8,7.5]), np.array([0,10,18,20]))))
    elif band =='R':
        type=int(round(np.interp(absmag, np.array([3.5-.33,5.1-.47,6.8-.8,7.5-.97]), np.array([0,10,18,20]))))
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
    
    ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),interval=interval, repeat=False)
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


def build_proximity_graph(ra, dec, overlap_sep):
    """
    Generate a dict of lists for crappy_min_vertex_cover_cut
    with edges between points closer than overlap_sep degrees
    
    ra and dec are arrays of coordinates in degrees
    """
    edges={}
    for node, cord in enumerate(zip(ra,dec)):
        seps=dist_radec_fast(cord[0], cord[1], ra, dec,
                             method='Haversine', unit='deg',
                             scale=overlap_sep)
        neighbors=np.where( seps < overlap_sep )[0].tolist()
        neighbors.remove(node) #no connection to yourself
        if len(neighbors)!=0:
            edges[node]=neighbors
    return edges

#Part B
#Now for each star with a conflict there are a few cases
# Ideally we would find the (potentially weighted) minimal vertex cover
#  and drop it, but that is getting all fancy and graph theoretic
def crappy_min_vertex_cover_cut(n_nodes,edgegraph, ID='graph'):
    """
    Takes the number of nodes in the graph and a dict of lists
    keys into dict and node ids and the lists contain the ids of nodes
    to which edges exist
    
    lower node ids are given priority if a pair is isolated
    
    Returns the nodes in the disconnected graph
    """
    nodes=range(n_nodes)
    from collections import OrderedDict
    edgegraph=OrderedDict(edgegraph)
    
    single_drop_count=0
    multi_drop_count=0
    try:
        while len(edgegraph) > 0:
            node,edge_set=edgegraph.popitem(last=False)
            #Case one, a pair of conflicting targets
            if len(edge_set)==1 and len(edgegraph[edge_set[0]])==1:
                assert node in edgegraph[edge_set[0]]
                #Drop the lower ranked of node and edges[node][0]
                # and remove them from the graph
                single_drop_count+=1
                nodes.remove(node if node > edge_set[0] else edge_set[0])
                edgegraph.pop(edge_set[0])
            #Case 2, a set of 3 or more conflicting targets
            else:
                if len(edge_set)>1:
                    multi_drop_count+=1
                    nodes.remove(node)
                    for node2 in edge_set:
                        if len(edgegraph[node2]) == 1:
                            edgegraph.pop(node2)
                        else:
                            edgegraph[node2].remove(node)
                else:
                    edgegraph[node]=edge_set
    except Exception, e:
        print str(e)
        import pdb;pdb.set_trace()
    
    if multi_drop_count >0:
        print "Warning: Dropped {} multi-overlapping stars in {}".format(
                    multi_drop_count,ID)
    
    return nodes

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
