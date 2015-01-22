import math

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
    """ Convert decimal degrees to h, m, s or d, m ,s
    slips seconds at the .1uas level
    """
    if type(n) not in (float, int):
        raise ValueError('Must give d as float or int')
    
    sign=-1.0 if n < 0 else 1.0
    n=abs(float(n))
    if ra:
        n/=15.0
#        hord=int(n)
#        m=int((n-hord)*60)
#        secs=(n-hord)*3600-m*60
#
#    else:

    hord=int(n)
    m=int((n-hord)*60)
    if m >=60:
        hord+=m/60
        m-=m - m % 60
    secs=(n-hord)*3600-m*60

#    import ipdb;ipdb.set_trace()
    if 60-secs < .00001: secs=60.0

    if secs>=60:
        isec=int(secs)
        m+=isec/60
        secs-=isec - isec % 60

    if m >=60:
        hord+=m/60
        m-=m - m % 60

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
    """convert a sexgesmal number to something"""
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
            import pdb;pdb.set_trace()
    test_inner(True,
               ['11:12:09.85','-00:34:32.02','00:34:32.02','-10:34:32.02'],
               [15*(11+12.0/60+9.85/3600), -15*(0+34.0/60+32.02/3600),
                15*(0+34.0/60+32.02/3600), -15*(10+34.0/60+32.02/3600)])

    test_inner(False,
               ['71:12:09.85','-00:34:32.02','00:34:32.02','-80:34:32.02'],
               [(71+12.0/60+9.85/3600), -(0+34.0/60+32.02/3600),
                (0+34.0/60+32.02/3600), -(80+34.0/60+32.02/3600)])

test_sexconvert()

def roundTo(x, value):
    """ Round to the nearest value """
    return int(round(x/value))*value

def dm2d(dm):
    return int(round(10.0**((dm+5.0)/5.0)))

def d2dm(parsec):
    return -5 + 5.0*np.log10(parsec)
