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

fluxdata={
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

def query_skycalc(ra, dec, utc, obs_lat_deg, obs_lon_deg, outfile):

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
    night_third=ceil((float(night_elapsed.seconds)/night_length.seconds) * 3.0)

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

    br['SKYMODEL.WAVELENGTH.MIN']='650'
    br['SKYMODEL.WAVELENGTH.MAX']='750'
    br['SKYMODEL.WAVELENGTH.GRID.MODE']=['fixed_spectral_resolution']
    br['SKYMODEL.WAVELENGTH.RESOLUTION']='500000'
    br['SKYMODEL.LSF.KERNEL.TYPE']=['none']

    print('Submitting form')
    br.submit()
    print('Downloading')
    br.retrieve(br.find_link(text='skytable.fits').absolute_url, outfile)

    br.close()
    print('Done')


#2516 telluric
im_keys=['b586-591.fits.gz',
'b1242-1245.fits.gz',
'feb_b869-871.fits.gz',

#2516
'b118-128.fits.gz',
'b408-417.fits.gz',
'b723-728, 730.fits.gz',
'b795-797, 799-803.fits.gz',
'b1065-1070.fits.gz',
'feb_b28, 31-35.fits.gz',
'feb_b128-132.fits.gz',
'feb_b537-541.fits.gz',

#hip
'b136-138.fits.gz',
'b254-256.fits.gz',
'b0598.fits.gz',
'b733-734.fits.gz',
'b811-813.fits.gz',
'b1006-1007.fits.gz',
'b1075-1077.fits.gz',
'b1258-1259.fits.gz',

'r132, 134-135.fits.gz',
'r251-253.fits.gz',
'r594-597.fits.gz',
'r735-737.fits.gz',
'r806-809.fits.gz',
'r1001-1005.fits.gz',
'r1072-1074.fits.gz',
'r1256-1257.fits.gz',

'feb_r38-40.fits.gz',
'feb_b135-137.fits.gz',
'feb_r272-274.fits.gz',
'feb_r544-546.fits.gz',
'feb_b1190-1192.fits.gz',

#2422
'b1248-1253.fits.gz',
'feb_b265-269.fits.gz',
'feb_b383-386.fits.gz',
'feb_b642-645.fits.gz',
'feb_b999-1003.fits.gz']



obs_lat_deg=np.rad2deg(-0.5063938434309143)
obs_lon_deg=np.rad2deg(-1.2338154852026897)
for k in im_keys:
    im=SURVEY_IMAGES[k]
    outfile='skys/skycalc_{}_vac.fits'.format(os.path.basename(
                      im.file).split('.')[0][1:])
    ra,dec=im.info.header['RA'], im.info.header['DEC']

    utc=im.midpoint.datetime

    query_skycalc(ra, dec, utc, obs_lat_deg, obs_lon_deg, outfile)







