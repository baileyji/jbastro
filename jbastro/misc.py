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

