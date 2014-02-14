import numpy as np
import matplotlib.pyplot as plt

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

def gaussfit(xdata, ydata):
    import numpy as np
    from scipy.optimize import curve_fit

    def fit_func(x, a0, a1, a2, a3, a4, a5):
        z = (x - a1) / a2
        y = a0 * np.exp(-z**2 / a2) + a3 + a4 * x + a5 * x**2
        return y

    parameters, covariance = curve_fit(fit_func, xdata, ydata)

    return parameters, covariance

