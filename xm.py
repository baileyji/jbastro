#!/bin/env python2.7
import numpy as np
from astropy.io import ascii
from astropy.io import fits
from astropy import coordinates as coord
from astropy.coordinates.angles import RA
from astropy.coordinates.angles import Dec
from astropy import units as u
from queryVizier import send_query_vizier
from great_circle_dist import dist_radec_fast, dist_radec
from field import Field
import os.path
import itertools
import types
from astroLib import *
import NewKC as KC

XMATCH_SCALE_DEG=2./3600

def xmatch(data, stars,summary=False,verbose=False):

    ra=stars['RAJ2000']
    de=stars['DEJ2000']

    data2stars=np.zeros(len(data))
    haveXM=np.zeros(len(data),dtype=np.bool)

    skipped=0
    nomatch=0
    for i,s in enumerate(data):

        seps=dist_radec_fast(s['RAJ2000'], s['DEJ2000'],
                        ra, de,
                        method='Haversine', unit='deg',
                        scale=XMATCH_SCALE_DEG)
        match,=np.where(seps<XMATCH_SCALE_DEG)
        nmatch=len(match)
        if nmatch==0:
            if verbose:
                print "No match for {}, V={:.1f}".format(s['ID'],s['Vmag'])
            nomatch+=1
        elif nmatch==1:
            data2stars[i]=match[0]
            haveXM[i]=True
        else:
            if verbose:
                print "{} matches for {}".format(nmatch, s['ID'])
            skipped+=1

    if summary:
        print "Xmatch complete"
        print "Skipped {} of {} data stars due to multiple matches".format(
            skipped, len(data))
        print "No matches for {} of {} data stars".format(nomatch, len(data))
    return (haveXM, data2stars)
