import mechanize
from BeautifulSoup import MinimalSoup
import ephem
import numpy as np

class PrettifyHandler(mechanize.BaseHandler):
    def http_response(self, request, response):
        if not hasattr(response, "seek"):
            response = mechanize.response_seek_wrapper(response)
        # only use BeautifulSoup if response is html
        if (response.info().dict.has_key('content-type') and
            ('html' in response.info().dict['content-type'])):
            soup = MinimalSoup (response.get_data())
            response.set_data(soup.prettify())
        return response

#http://www.solarham.net/averages.htm
fluxdata={
    ('September',2016):  87.5,
    ('August'   ,2016):  85.0,
    ('July'     ,2016):  85.9,
    ('June'     ,2016):  81.9,
    ('May'      ,2016):  93.1,
    ('April'    ,2016):  93.4,
    ('March'    ,2016):  91.6,
    ('February' ,2016): 103.5,
    ('January'  ,2016): 103.5,
    ('December' ,2015): 112.8,
    ('November' ,2015): 109.6,
    ('October'  ,2015): 104.1,
    ('September',2015): 102.1,
    ('August'   ,2015): 106.2,
    ('July'     ,2015): 107.0,
    ('June'     ,2015): 123.2,
    ('May'      ,2015): 120.1,
    ('April'    ,2015): 129.2,
    ('March'    ,2015): 126.0,
    ('February' ,2015): 128.8,
    ('January'  ,2015): 141.7,
    ('December' ,2014): 158.7,
    ('November' ,2014): 155.2,
    ('October'  ,2014): 153.7,
    ('September',2014): 146.1,
    ('August'   ,2014): 124.7,
    ('July'     ,2014): 137.3,
    ('June'     ,2014): 122.2,
    ('May'      ,2014): 130.0,
    ('April'    ,2014): 144.3,
    ('March'    ,2014): 149.9,
    ('February' ,2014): 170.3,
    ('January'  ,2014): 158.6,
    ('December' ,2013): 147.7,
    ('November' ,2013): 148.4,
    ('October'  ,2013): 132.3,
    ('September',2013): 102.7,
    ('August'   ,2013): 114.7,
    ('July'     ,2013): 115.6,
    ('June'     ,2013): 110.2,
    ('May'      ,2013): 131.3,
    ('April'    ,2013): 125.0,
    ('March'    ,2013): 111.2,
    ('February' ,2013): 104.4,
    ('January'  ,2013): 127.1,
    ('December' ,2012): 108.4,
    ('November' ,2012): 120.9,
    ('October'  ,2012): 123.3,
    ('September',2012): 123.2,
    ('August'   ,2012): 115.7,
    ('July'     ,2012): 135.6,
    ('June'     ,2012): 120.5,
    ('May'      ,2012): 121.5,
    ('April'    ,2012): 113.1,
    ('March'    ,2012): 115.1,
    ('February' ,2012): 106.8,
    ('January'  ,2012): 133.1,
    ('December' ,2011): 141.3,
    ('November' ,2011): 153.1,
    ('October'  ,2011): 137.2,
    ('September',2011): 134.5,
    ('August'   ,2011): 101.7,
    ('July'     ,2011):  94.2,
    ('June'     ,2011):  95.8,
    ('May'      ,2011):  95.9,
    ('April'    ,2011): 112.6,
    ('March'    ,2011): 115.3,
    ('February' ,2011):  94.5,
    ('January'  ,2011):  83.7,
    ('December' ,2010):  84.3}

def query_skycalc(ra, dec, utc, obs_lat_deg, obs_lon_deg, outfile,
                   wmin=650, wmax=750, debug=False):

    #compute Ephemeris data
    targ=ephem.Equatorial(ra, dec)
    ecl=ephem.Ecliptic(targ)

    obs=ephem.Observer()
    #obs.date=ephem.julian_date(o.image.midpoint.jd) #This does not work
    obs.date=utc #This is correct, ephem wants UTC
    obs.lat=ephem.degrees('{}'.format(obs_lat_deg))
    obs.long=ephem.degrees('{}'.format(obs_lon_deg))
    #obs.elevation=2450.0

    star=ephem.FixedBody()
    star._ra=ephem.hours(ra)
    star._dec=ephem.degrees(dec)
    star.compute(obs)

    moon=ephem.Moon(obs)
    sun=ephem.Sun(obs)


    helio_ecl_lon=np.rad2deg(ecl.lon)
    #This is a guess, it could be that I've got the signs flipped
    if helio_ecl_lon >180: helio_ecl_lon=-(360-helio_ecl_lon)
    ecl_lat=np.rad2deg(ecl.lat)


    targ_alt=np.rad2deg(star.alt)
    targ_X=1.0/np.cos(np.deg2rad(90-targ_alt))

    moon_alt=np.rad2deg(moon.alt)
    moon_targ_sep=np.rad2deg(ephem.separation(moon, star))
    sum_moon_sep=np.rad2deg(ephem.separation(moon, sun))


    month_pair=(obs.date.datetime().month%12)/2 + 1

    night_length=(obs.next_rising(sun).datetime()-
                 obs.previous_setting(sun).datetime())
    night_elapsed=obs.date.datetime()-obs.previous_setting(sun).datetime()

    # 1,2, 3
    night_third=np.ceil((float(night_elapsed.seconds)/night_length.seconds) * 3.0)

    try:
        solar_flux=fluxdata[(obs.date.datetime().strftime('%B'),
                             obs.date.datetime().year)]
    except KeyError,e:
        print str(e)
        raise ValueError('No mean solar flux for observation date')


    #Connect to skycalc and query
    br = mechanize.Browser()
    br.add_handler(PrettifyHandler())
    
    #ESO skycalc constraint
    do_moon=(moon_alt-targ_alt) <= moon_targ_sep <= (180 - targ_alt - moon_alt)


    print('Opening Connection to eso.org')
    br.open('http://www.eso.org/observing/etc/bin/'
            'gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC')


    br.select_form(name='form1')

    br['SKYMODEL.TARGET.ALT']='{}'.format(targ_alt)
    br['SKYMODEL.TARGET.AIRMASS']='{}'.format(targ_X)


    br['SKYMODEL.SEASON']=['{}'.format(month_pair)]
    br['SKYMODEL.TIME']=['{}'.format(int(night_third))]
    br['SKYMODEL.PWV.MODE']=['season']
    br['SKYMODEL.MSOLFLUX']='{}'.format(solar_flux)


    br.find_control('SKYMODEL.INCL.MOON').items[0].selected=do_moon
    br['SKYMODEL.MOON.SUN.SEP']='{}'.format(sum_moon_sep)
    br['SKYMODEL.MOON.TARGET.SEP']='{}'.format(moon_targ_sep)
    br['SKYMODEL.MOON.ALT']='{}'.format(moon_alt)
    br['SKYMODEL.MOON.EARTH.DIST']='1.0000'
    br['SKYMODEL.ECL.LON']='{}'.format(helio_ecl_lon)
    br['SKYMODEL.ECL.LAT']='{}'.format(ecl_lat)

    br.find_control('SKYMODEL.INCL.STARLIGHT').items[0].selected=True
    br.find_control('SKYMODEL.INCL.ZODIACAL').items[0].selected=True
    br.find_control('SKYMODEL.INCL.MOLEC.EMIS.LOWER.ATM').items[0].selected=True
    br.find_control('SKYMODEL.INCL.MOLEC.EMIS.UPPER.ATM').items[0].selected=True
    br.find_control('SKYMODEL.INCL.AIRGLOW').items[0].selected=True

    br.find_control('SKYMODEL.INCL.THERMAL').items[0].selected=False
    br.find_control('SKYCALC.RAD.PLOT.FLAG').items[0].selected=False
    br.find_control('SKYCALC.TRANS.PLOT.FLAG').items[0].selected=False
    br.find_control('SKYCALC.MAG.FLAG').items[0].selected=False
    br.find_control('SKYCALC.LSF.PLOT.FLAG').items[0].selected=False

    br['SKYMODEL.WAVELENGTH.MIN']='{:.0f}'.format(wmin)
    br['SKYMODEL.WAVELENGTH.MAX']='{:.0f}'.format(wmax)
    br['SKYMODEL.WAVELENGTH.GRID.MODE']=['fixed_spectral_resolution']
    br['SKYMODEL.WAVELENGTH.RESOLUTION']='500000'
    br['SKYMODEL.LSF.KERNEL.TYPE']=['none']

    if debug:
        
        print 'SKYMODEL.TARGET.ALT={}'.format(targ_alt)
        print 'SKYMODEL.TARGET.AIRMASS={}'.format(targ_X)
        
        
        print 'SKYMODEL.SEASON={}'.format(month_pair)
        print 'SKYMODEL.TIME={}'.format(int(night_third))
        print 'SKYMODEL.PWV.MODE=season'
        print 'SKYMODEL.MSOLFLUX={}'.format(solar_flux)
        
        
        print 'SKYMODEL.INCL.MOON={}'.format(do_moon)
        print 'SKYMODEL.MOON.SUN.SEP={}'.format(sum_moon_sep)
        print 'SKYMODEL.MOON.TARGET.SEP={}'.format(moon_targ_sep)
        print 'SKYMODEL.MOON.ALT={}'.format(moon_alt)
        print 'SKYMODEL.MOON.EARTH.DIST=1.0000'
        print 'SKYMODEL.ECL.LON={}'.format(helio_ecl_lon)
        print 'SKYMODEL.ECL.LAT={}'.format(ecl_lat)
        
        print 'SKYMODEL.INCL.STARLIGHT=True'
        print 'SKYMODEL.INCL.ZODIACAL=True'
        print 'SKYMODEL.INCL.MOLEC.EMIS.LOWER.ATM=True'
        print 'SKYMODEL.INCL.MOLEC.EMIS.UPPER.ATM=True'
        print 'SKYMODEL.INCL.AIRGLOW=True'
        
        print 'SKYMODEL.INCL.THERMAL=False'
        print 'SKYCALC.RAD.PLOT.FLAG=False'
        print 'SKYCALC.TRANS.PLOT.FLAG=False'
        print 'SKYCALC.MAG.FLAG=False'
        print 'SKYCALC.LSF.PLOT.FLAG=False'
        
        print 'SKYMODEL.WAVELENGTH.MIN=650'
        print 'SKYMODEL.WAVELENGTH.MAX=750'
        print 'SKYMODEL.WAVELENGTH.GRID.MODE=fixed_spectral_resolution'
        print 'SKYMODEL.WAVELENGTH.RESOLUTION=500000'
        print 'SKYMODEL.LSF.KERNEL.TYPE=none'

    print('Submitting form')
    br.submit()
    print('Downloading')
    br.retrieve(br.find_link(text='skytable.fits').absolute_url, outfile)

    br.close()
    print('Done')

