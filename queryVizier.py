import urllib
DEFAULT_FILENAME='vizier'
#shameless rip from aspylib
def send_query_vizier(catalog, constraint, position=None, conesize=None,
                      filename = DEFAULT_FILENAME, format='fits-bin'):
    """
        ---------------------
        Purpose
        Sends a query to Vizier and retrieve a list of stars with astrometric (alpha, delta, etc) or photometric (magnitudes, etc) information
        The data is saved in a specific xml file, with VOtable format
        The syntax to send a Vizier query is described there: http://vizier.u-strasbg.fr/doc/asu-summary.htx
        ---------------------
        Inputs
        * catalog (string) = name of the catalog, following Vizier syntax
        Popular Vizier catalogs include:
        'II/294' - Sloan SDSS photometric catalog Release 7 (2009)
        '2MASS-PSC' - 2MASS point source catalog (2003)
        'GSC2.3' - Version 2.3.2 of the HST Guide Star Catalog (2006)
        'USNO-B1' - Verson B1 of the US Naval Observatory catalog (2003)
        'NVSS'  - NRAO VLA Sky Survey (1998)
        'B/DENIS/DENIS' - 2nd Deep Near Infrared Survey of southern Sky
        'I/259/TYC2' - Tycho-2 main catalog (2000)
        'I/311/HIP2' - Hipparcos main catalog, new reduction (2007)	Each string gives the complete filename (including path) of a file that contains the lightcurve measurements.
        * position (string or [float,float]) = position name (resolved by Sesame) or alpha/delta position (in degrees) giving the center of the star field to be requested
        * conesize (None, float or [float,float]) = size in arcmin, corresponds to the radius (float) or alpha/delta box rectangular size ([float,float]) defining the star field requested
        If equal to None (default), a radius of 5 arcmin is used
        * constraint (string) = list of constraints to be applied on the catalogs fields to filter the retrieved stars (Vizier syntax)
        For instance: 'B1mag<=13'
        Multiple contraints are separated by comma: 'B1mag<13,B2mag<14'
        * filename (string) = optional parameter, default is 'vizier.xml'.
        Specifies the xml file in which the data from the Vizier request is stored.
        It can be a full filename with path information, or a simple filename in which case the active directory is used.
        ---------------------
        Output = no output
        ---------------------
    """
    cat = "?-source=" + catalog
    if position!=None:
        if isinstance(position,str):
            obj = "&-c=" + position
        else:
            obj = "&-c.ra=" + str(position[0]) + "&-c.dec=" + str(position[1])
        if conesize==None:
            dis = "&-c.rm=05"
        elif isinstance(conesize,list):
            dis = "&-c.bm=" + str(int(conesize[0])) + "/" + str(int(conesize[1]))
        else:
            dis = "&-c.rd={:.3}".format(float(conesize))
    else:
        obj=''
        dis=''
    if len(constraint)==0:
        cons = ""
    else:
        constraint = constraint.replace(" ","")
        constraint = constraint.replace(",","&")
        constraint = constraint.replace("<","=%3C")
        constraint = constraint.replace(">","=%3E")
        constraint = constraint.replace("+","%2B")
        constraint = constraint.replace("/","%2F")
        constraint = constraint.replace("!","=!")
        cons = "&" + constraint
    if format=='votable':
        fmt='votable/'
        ext='.xml'
    elif format=='fits-ascii':
        fmt='asu-fits'
        ext='.fits'
    elif format=='fits-bin':
        fmt='asu-binfits'
        ext='.fits'
    else:
        raise Exception('Format must be votable, fits-bin, or fits-ascii')
    url = "http://webviz.u-strasbg.fr/viz-bin/"+fmt + cat + obj + dis + cons + "&-out.all&-out.max=unlimited"
    print url
    if filename==DEFAULT_FILENAME:
        filename+=ext

    fp = urllib.urlopen(url)

    op = open(filename, "wb")
    
    while 1:
        s = fp.read(8192)
        if not s:
            break
        op.write(s)
    fp.close()
    op.close()
        