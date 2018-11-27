import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from .robust import *
import copy

def rangify(data,delim=', '):
    """Stackoverflow"""
    from itertools import groupby
    from operator import itemgetter
    str_list = []
    data = list(data)
    data.sort()
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        ilist = map(itemgetter(1), g)
        if len(ilist) > 1:
            str_list.append('%d-%d' % (ilist[0], ilist[-1]))
        else:
            str_list.append('%d' % ilist[0])
    return delim.join(str_list)

def derangify(s,delim=','):
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
    for x in s.split(delim):
        t=x.split('-')
        if len(t) not in [1,2]:
            raise SyntaxError("'{}'".format(s)+
                              "does not seem to be derangeifyable")
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

def normspec(sin, doplot=False, min_good_frac=.05, poly_pow=7, maskin=None,
             sigmau=2.0, sigmal=1.0, region=None, med_wid=3, maxreps=5,
             robust=True):
    """
    Normalize a spectrum
    
    Fit a polynomial, find the bits within sigmal and sigmau of the
    curve, repeat. quit when less than min_good_frac of the points are left
    in the data.
    """
    
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


def stddev_bias_corr(n):
    if n == 1:
        corr=1.0
    else:
        lut=[0.7978845608,0.8862269255,0.9213177319,0.9399856030,0.9515328619,
             0.9593687891,0.9650304561,0.9693106998,0.9726592741,1.0]
        lut_ndx=max(min(n-2,len(lut)-1),0)
        
        corr=lut[lut_ndx]

    return 1.0/corr

def mswd(x,w,axis=None):
    """
    http://en.wikipedia.org/wiki/Mean_square_weighted_deviation
    """
    wsum=w.sum(axis)
    xm=(w*x).sum(axis)/wsum
    return xm, np.sqrt(wsum/(wsum**2-(w**2).sum(axis))*(w*(x-xm)**2).sum(axis))


def find_prob_contours(kde_p,dxdy, pvals=(.1,.333,.5,.666,.9)):

    from scipy.optimize import brentq

    peak = kde_p.max()

    max_pval=kde_p.sum() * dxdy

    # You can't quite do this since the function is discontinuous, and never
    # actually equals 0, but you can do a search along these lines.
    # Exercise left for the reader.
    def f(prob):
        return lambda x: prob - (kde_p[kde_p > x].sum() * dxdy)

    try:
        return tuple([(brentq(f(p), 0, peak),p) for p in pvals if p< max_pval])
    except ValueError:
        import ipdb;ipdb.set_trace()

def kden_plot(X,Y, im, support, pvals=(.1,.333,.5,.666,.9),
              contours=True, imshow=True, cbar=False, psupport=True,oned=True,
              fignum=None,cfontsize=10):
    if fignum is None:
        fignum=plt.gcf().number
        plt.clf()
    
    dx=np.abs(X[0,0]-X[1,0])
    dy=np.abs(Y[0,0]-Y[0,1])
    dxdy=dx*dy
    if contours:
        
        levelsdict={x:'{:3.0f}%'.format(p*100)
                    for x,p in find_prob_contours(im,dxdy,pvals=pvals)}
        cplt=plt.contour(X,Y,im,colors='k', levels=levelsdict.keys())
        plt.clabel(cplt, inline=1, fontsize=10, use_clabeltext=True,
                   fmt=levelsdict)
    if imshow:
        plt.imshow(np.rot90(im),aspect='auto',
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   cmap=plt.cm.gist_earth_r)
    if cbar:
        plt.colorbar()

    if psupport:
        plt.plot(support[0],support[1],'r,')

    if oned:
        plt.figure(fignum+1)
        plt.plot(X[:,0],((im*dy).sum(1)))
        #add shading for shadecore

arange=np.arange
import math
log=np.log
import warnings

def _anderson_ksamp_midrank(samples, Z, Zstar, k, n, N):
    """
    Compute A2akN equation 7 of Scholz and Stephens.
    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.
    Returns
    -------
    A2aKN : float
        The A2aKN statistics of Scholz and Stephens 1987.
    """

    A2akN = 0.
    Z_ssorted_left = Z.searchsorted(Zstar, 'left')
    if N == Zstar.size:
        lj = 1.
    else:
        lj = Z.searchsorted(Zstar, 'right') - Z_ssorted_left
    Bj = Z_ssorted_left + lj / 2.
    for i in arange(0, k):
        s = np.sort(samples[i])
        s_ssorted_right = s.searchsorted(Zstar, side='right')
        Mij = s_ssorted_right.astype(np.float)
        fij = s_ssorted_right - s.searchsorted(Zstar, 'left')
        Mij -= fij / 2.
        inner = lj / float(N) * (N * Mij - Bj * n[i])**2 / \
            (Bj * (N - Bj) - N * lj / 4.)
        A2akN += inner.sum() / n[i]
    A2akN *= (N - 1.) / N
    return A2akN


def _anderson_ksamp_right(samples, Z, Zstar, k, n, N):
    """
    Compute A2akN equation 6 of Scholz & Stephens.
    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.
    Returns
    -------
    A2KN : float
        The A2KN statistics of Scholz and Stephens 1987.
    """

    A2kN = 0.
    lj = Z.searchsorted(Zstar[:-1], 'right') - Z.searchsorted(Zstar[:-1],
                                                              'left')
    Bj = lj.cumsum()
    for i in arange(0, k):
        s = np.sort(samples[i])
        Mij = s.searchsorted(Zstar[:-1], side='right')
        inner = lj / float(N) * (N * Mij - Bj * n[i])**2 / (Bj * (N - Bj))
        A2kN += inner.sum() / n[i]
    return A2kN

def anderson_ksamp(samples, midrank=True):
    """The Anderson-Darling test for k-samples.
    The k-sample Anderson-Darling test is a modification of the
    one-sample Anderson-Darling test. It tests the null hypothesis
    that k-samples are drawn from the same population without having
    to specify the distribution function of that population. The
    critical values depend on the number of samples.
    Taken from scipy v0.14 source
    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample data in arrays.
    midrank : bool, optional
        Type of Anderson-Darling test which is computed. Default
        (True) is the midrank test applicable to continuous and
        discrete populations. If False, the right side empirical
        distribution is used.
    Returns
    -------
    A2 : float
        Normalized k-sample Anderson-Darling test statistic.
    critical : array
        The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%.
    p : float
        An approximate significance level at which the null hypothesis for the
        provided samples can be rejected.
    Raises
    ------
    ValueError
        If less than 2 samples are provided, a sample is empty, or no
        distinct observations are in the samples.
    See Also
    --------
    ks_2samp : 2 sample Kolmogorov-Smirnov test
    anderson : 1 sample Anderson-Darling test
    Notes
    -----
    [1]_ Defines three versions of the k-sample Anderson-Darling test:
    one for continuous distributions and two for discrete
    distributions, in which ties between samples may occur. The
    default of this routine is to compute the version based on the
    midrank empirical distribution function. This test is applicable
    to continuous and discrete data. If midrank is set to False, the
    right side empirical distribution is used for a test for discrete
    data. According to [1]_, the two discrete test statistics differ
    only slightly if a few collisions due to round-off errors occur in
    the test not adjusted for ties between samples.
    .. versionadded:: 0.14.0
    References
    ----------
    .. [1] Scholz, F. W and Stephens, M. A. (1987), K-Sample
           Anderson-Darling Tests, Journal of the American Statistical
           Association, Vol. 82, pp. 918-924.
    Examples:
    ---------
    >>> from scipy import stats
    >>> np.random.seed(314159)
    The null hypothesis that the two random samples come from the same
    distribution can be rejected at the 5% level because the returned
    test value is greater than the critical value for 5% (1.961) but
    not at the 2.5% level. The interpolation gives an approximate
    significance level of 3.1%:
    >>> stats.anderson_ksamp([np.random.normal(size=50),
    ... np.random.normal(loc=0.5, size=30)])
    (2.4615796189876105,
      array([ 0.325,  1.226,  1.961,  2.718,  3.752]),
      0.03134990135800783)
    The null hypothesis cannot be rejected for three samples from an
    identical distribution. The approximate p-value (87%) has to be
    computed by extrapolation and may not be very accurate:
    >>> stats.anderson_ksamp([np.random.normal(size=50),
    ... np.random.normal(size=30), np.random.normal(size=20)])
    (-0.73091722665244196,
      array([ 0.44925884,  1.3052767 ,  1.9434184 ,  2.57696569,  3.41634856]),
      0.8789283903979661)
    """

    k = len(samples)
    if (k < 2):
        raise ValueError("anderson_ksamp needs at least two samples")

    samples = list(map(np.asarray, samples))
    Z = np.sort(np.hstack(samples))
    N = Z.size
    Zstar = np.unique(Z)
    if Zstar.size < 2:
        raise ValueError("anderson_ksamp needs more than one distinct "
                         "observation")

    n = np.array([sample.size for sample in samples])
    if any(n == 0):
        raise ValueError("anderson_ksamp encountered sample without "
                         "observations")

    if midrank:
        A2kN = _anderson_ksamp_midrank(samples, Z, Zstar, k, n, N)
    else:
        A2kN = _anderson_ksamp_right(samples, Z, Zstar, k, n, N)

    h = (1. / arange(1, N)).sum()
    H = (1. / n).sum()
    g = 0
    for l in arange(1, N-1):
        inner = np.array([1. / ((N - l) * m) for m in arange(l+1, N)])
        g += inner.sum()

    a = (4*g - 6) * (k - 1) + (10 - 6*g)*H
    b = (2*g - 4)*k**2 + 8*h*k + (2*g - 14*h - 4)*H - 8*h + 4*g - 6
    c = (6*h + 2*g - 2)*k**2 + (4*h - 4*g + 6)*k + (2*h - 6)*H + 4*h
    d = (2*h + 6)*k**2 - 4*h*k
    sigmasq = (a*N**3 + b*N**2 + c*N + d) / ((N - 1.) * (N - 2.) * (N - 3.))
    m = k - 1
    A2 = (A2kN - m) / math.sqrt(sigmasq)

    # The b_i values are the interpolation coefficients from Table 2
    # of Scholz and Stephens 1987
    b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326])
    b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822])
    b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396])
    critical = b0 + b1 / math.sqrt(m) + b2 / m
    pf = np.polyfit(critical, log(np.array([0.25, 0.1, 0.05, 0.025, 0.01])), 2)
    if A2 < critical.min() or A2 > critical.max():
        warnings.warn("approximate p-value will be computed by extrapolation")

    p = math.exp(np.polyval(pf, A2))
    return A2, critical, p




#Put this here so it isn't lost forever (taken from another file elsewhere)
#def imsurffit(im, order=3):
#    """ ignore cross terms"""
#    
#    xmin,xmax=0,im.shape[1]-1
#    ymin,ymax=0,im.shape[0]-1
#    xx, yy = np.meshgrid(np.arange(xmin,xmax+1), np.arange(ymin,ymax+1))
#    
#    m = polyfit2d(xx.ravel(),yy.ravel(),im.ravel(),order=order)
#    
#    # Evaluate it
#    zz = polyval2d(xx, yy, m)
#    return zz,m
#
#
#def polyfit2d(x, y, z, order=3):
#    import itertools
#    ncols = (order + 1)**2
#    G = np.zeros((x.size, ncols))
#    ij = itertools.product(range(order+1), range(order+1))
#    for k, (i,j) in enumerate(ij):
#        G[:,k] = x**i * y**j
#    m, _, _, _ = np.linalg.lstsq(G, z)
#    return m
#
#def polyval2d(x, y, m):
#    import itertools
#    order = int(np.sqrt(len(m))) - 1
#    ij = itertools.product(range(order+1), range(order+1))
#    z = np.zeros_like(x)
#    for a, (i,j) in zip(m, ij):
#        z += a * x**i * y**j
#    return z


