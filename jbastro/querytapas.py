import mechanize
import numpy as np

UNAME='baileyji@umich.edu'
UPASS='tapas_jbailey'

_atm_str='{%22reference%22:{%22firstValue%22:%220,1,2,3,4,5,6%22,%22secondValue%22:0}}'

_pref_str=('{'
'%22format%22:{%22firstValue%22:%22ASCII,FITS,NETCDF,VO%22,%22secondValue%22:%22FITS%22},'
'%22rayleighExtinction%22:{%22firstValue%22:%22YES,NO%22,%22secondValue%22:%22YES%22},'
'%22h2oExtinction%22:{%22firstValue%22:%22YES,NO%22,%22secondValue%22:%22YES%22},'
'%22o3Extinction%22:{%22firstValue%22:%22YES,NO%22,%22secondValue%22:%22YES%22},'
'%22o2Extinction%22:{%22firstValue%22:%22YES,NO%22,%22secondValue%22:%22YES%22},'
'%22co2Extinction%22:{%22firstValue%22:%22YES,NO%22,%22secondValue%22:%22YES%22},'
'%22ch4Extinction%22:{%22firstValue%22:%22YES,NO%22,%22secondValue%22:%22YES%22},'
'%22n2oExtinction%22:{%22firstValue%22:%22YES,NO%22,%22secondValue%22:%22YES%22},'
'%22bervCorrection%22:{%22firstValue%22:%22YES,NO%22,%22secondValue%22:%22NO%22}}')


_url=('http://ether.ipsl.jussieu.fr/tapas/project?methodName=login&login='+
      UNAME+'&pwd='+UPASS)
def query_tapas(utc_str, zangle, wmin=650, wmax=750,Rval=800000,br=None):
    """
    utc_str in form '2013-11-22 08:51:00'
    zangle float of the zenith angle
    
    br optionally an open mechanize.Browser connection to TAPAS 
    
    return an opened mechanize.Browser connection to TAPAS
    """


    from time import mktime,strptime
    minw=str(int(wmin))
    maxw=str(int(wmax))
    zenithangle_str='{:.1f}'.format(zangle)

    R_str=str(int(Rval))

    #unix time with ms
    date_str=str(int(mktime(strptime(utc_str,'%Y-%m-%d %H:%M:%S'))*1000))

    obs_str=('{'
             '%22date%22:'+date_str+','
             '%22observatory%22:{%22name%22:%22Las%20Campanas%20Chile%22},'
             '%22los%22:{%22zenithAngle%22:%22'+zenithangle_str+'%22,'
                        '%22raJ2000%22:%22%22,%22decJ2000%22:%22%22},'
             '%22instrument%22:{'
                '%22ilsfChoice%22:{%22firstValue%22:%22-1,0,1%22,%22secondValue%22:-1},'
                '%22spectralChoice%22:{%22firstValue%22:%22NM_VACUUM,NM_STANDARD,CM%22,%22secondValue%22:%22NM_VACUUM%22},'
                '%22spectralRange%22:%22'+minw+'%20'+maxw+'%22,'
                '%22resolvingPower%22:{%22firstValue%22:%220%22,%22secondValue%22:%22'+R_str+'%22},'
                '%22samplingRatio%22:{%22firstValue%22:%220%22,%22secondValue%22:%220%22}}'
             '}')

    query=('http://ether.ipsl.jussieu.fr/tapas/data?methodName=createUserRequest&jsonTapas='
                       '{%22requests%22:'
                       '[{%22id%22:1,'
                       '%22preference%22:'+_pref_str+','
                       '%22observation%22:'+obs_str+','
                       '%22atmosphere%22:'+_atm_str+'}]}')

    if br is None:
        import mechanize
        br = mechanize.Browser()
        br.open(_url)
    
    print br.open(query).read()

    return br

x="""
2013-11-22 08:51 25.1
2013-11-22 09:01 23.2
2013-11-23 08:37 26.4
2013-11-23 08:46 24.8
2013-11-26 08:45 23.6
2013-11-26 08:54 21.7
2013-11-27 08:35 24.0
2013-11-27 08:45 22.8
2013-11-28 08:38 23.4
2013-11-28 08:54 20.9
2013-11-29 08:44 22.9
2013-11-29 08:56 20.0
2013-11-30 08:25 24.0
2013-11-30 08:36 22.3
2013-12-01 08:17 24.4
2013-12-01 08:25 23.2
2014-02-16 07:29 34.1
2014-02-17 06:17 22.6
2014-02-18 06:31 25.5
2014-02-21 06:39 28.8
2014-02-28 05:03 18.4
2014-12-09 08:20 19.4
2014-12-09 08:28 18.6
2014-12-09 08:39 17.2
2014-12-10 08:27 18.3
2014-12-10 08:46 17.0
2014-12-11 08:29 17.5
2014-12-12 08:28 17.0
2014-12-12 08:32 16.6
2014-12-13 08:42 15.5
2014-12-13 08:52 15.0
2014-12-17 08:45 14.8
2014-12-20 08:35 14.6
2014-12-20 08:46 14.6
2014-12-22 08:35 14.6"""
x=x.split('\n')[1:]

d,t,za=x[0].split()
d=d+' '+t+':00'
br=query_tapas(d,float(za),Rval=1000000)
for y in x[1:]:
    d,t,za=y.split()
    d=d+' '+t+':00'
    query_tapas(d,float(za),Rval=1000000,br=br)
