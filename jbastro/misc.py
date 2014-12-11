import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def rangify(data):
    """Stackoverflow"""
    from itertools import groupby
    from operator import itemgetter
    str_list = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        ilist = map(itemgetter(1), g)
        if len(ilist) > 1:
            str_list.append('%d-%d' % (ilist[0], ilist[-1]))
        else:
            str_list.append('%d' % ilist[0])
    return ', '.join(str_list)

def derangify(s):
    """
    Takes a range in form of "a-b" and generate a list of numbers between 
    a and b inclusive.
    Also accepts comma separated ranges like "a-b,c-d,f" will build a 
    list which will include
    Numbers from a to b, a to d and f
    http://code.activestate.com/recipes/577279-generate-list-of-
    numbers-from-hyphenated-and-comma/
    """
    s="".join(s.split())#removes white space
    r=set()
    for x in s.split(','):
        t=x.split('-')
        if len(t) not in [1,2]:
            raise SyntaxError("hash_range is given its "
                              "arguement as "+s+" which seems not "
                              "correctly formated.")
        if len(t)==1:
            r.add(int(t[0]))
        else:
            r.update(set(range(int(t[0]),int(t[1])+1)))
    l=list(r)
    l.sort()
    return tuple(l)


def share_memory(a, b):
    """Returns the number of shared bytes between arrays `a` and `b`."""
    #http://stackoverflow.com/a/11287440
    def byte_offset(a):
        """Returns a 1-d array of the byte offset of every element in `a`.
            Note that these will not in general be in order."""
        stride_offset = np.ix_(*map(range,a.shape))
        element_offset = sum(i*s for i, s in zip(stride_offset,a.strides))
        element_offset = np.asarray(element_offset).ravel()
        return np.concatenate([element_offset + x for x in range(a.itemsize)])
    a_low, a_high = np.byte_bounds(a)
    b_low, b_high = np.byte_bounds(b)
    
    beg, end = max(a_low,b_low), min(a_high,b_high)
    
    if end - beg > 0:
        # memory overlaps
        amem = a_low + byte_offset(a)
        bmem = b_low + byte_offset(b)
        
        return np.intersect1d(amem,bmem).size
    else:
        return 0


#;MUST NOT MODIFY   sin   as it will cause problems elsewhere (noticed 3/10/09 -JB)
#
#;Power (2) - polynomial order to use
#; min_good_frac (0.5) - quit when this fraction of spectrum is left
#; mask - bytarray(n_elements(s)) 1s indicate points in spectrum to omit from
#;  continuum search 
#;plot - shoe progress plots, must push enter to advance
#; coeff - polynomial coefficients of the norm to feed into poly()
#; Procedure: Fit a polynomial, find the bits above the curve, repeat
#; quit when less than min_good_frac of the points are left in the spectrum,
#; clearly this algorithm won't play well with emission lines
import scipy.signal
import numpy as np
from .robust import *
def normspec(sin, doplot=False, min_good_frac=.05, poly_pow=7, maskin=None,
             sigmau=2.0, sigmal=1.0, region=None, med_wid=3, maxreps=5,
             robust=True):
    w1=0 #;window 1
    w2=1 #;window 2
    
    sin=sin.copy()
	
	#;median smooth the input if requested
    gs = (scipy.signal.medfilt(sin, med_wid) if med_wid > 1 else sin)
    
    mask = np.zeros_like(gs, dtype=np.bool) | ~np.isfinite(gs)
    
    if type(maskin)!=type(None):
        mask|=maskin
    
    if type(region)!=type(None):
        mask[0:max(region[0],0)]=True
        mask[min(region[1],len(sin)-1)+1:]=True

    x=np.arange(len(sin))
    
    good=~mask
    n_init=(~mask).sum()
    
    done=False
    rep=0
    while not done:
        rep+=1
        try:
            if robust:
                c,yfit2,sig=ROBUST_POLY_FIT(x[good],gs[good],poly_pow)
            else:
                c=np.polyfit(x[good],gs[good],poly_pow)
                yfit2=np.poly1d(c)(x[good])
        except ValueError, e:
            import ipdb;ipdb.set_trace()
        normtrend=np.poly1d(c)(x)
        diff=(gs-normtrend)
        up_sig=sigmau*diff[good].std()
        lo_sig=sigmal*diff[good].std()
        good=(~mask) & (gs < normtrend+up_sig) & (gs > normtrend-lo_sig)
        ngood=good.sum()
        
        if doplot:
            plt.figure(w1)
            plt.plot(sin,'b')
            plt.plot(normtrend,'r')
            plt.plot(normtrend+up_sig,'g')
            plt.plot(normtrend-lo_sig,'g')
            plt.plot(np.where(~good)[0],sin[~good],'y,')

            plt.figure(w2)
            plt.plot(sin/normtrend)
            plt.ylim(0,2)
            plt.show(0)
            print('Good: {} ({:.0}%)'.format(ngood,100*float(ngood)/n_init))
            raw_input()
        
        if float(ngood)/n_init < min_good_frac:
            done=True
            if rep == 1:coeff=c
        else:
            coeff=c
            
        #;Some other finishing criterion as well?
        if rep > maxreps:
            #print('Terminating after {} iterations.'.format(rep)
            done=True

    normtrend=np.poly1d(coeff)(x)
    return sin/normtrend, coeff


def inpaint_mask(im_in,mask_in):
    ipn_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    im=im_in.copy()
    im[mask_in.astype(bool)]=np.nan
    nans = np.isnan(im)
    not_nans=~nans
    while np.sum(nans)>0:
        im[nans] = 0
        vNeighbors = scipy.signal.convolve2d(not_nans, ipn_kernel,
                                             mode='same', boundary='symm')
        im2 = scipy.signal.convolve2d(im, ipn_kernel,
                                      mode='same', boundary='symm')
        im2[vNeighbors>0] = im2[vNeighbors>0]/vNeighbors[vNeighbors>0]
        im2[vNeighbors==0] = np.nan
        im2[not_nans] = im[not_nans]
        im = im2
        nans = np.isnan(im)
        not_nans=~nans
    
    return im

>>>>>>> develop
