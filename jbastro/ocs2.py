def Teff_correction(Teff):
    import numpy as np
    (t0, dt, wt, b) = [6.21964405e+03, -4.15347433e+02, 5.36256660e-03, 1.24485027e+02]
    return dt / (1.0 + np.exp(-wt * (Teff - t0))) + b


def corrTeff(teff):
    return teff + Teff_correction(teff)


_jeffries_xray_report_targ = {}
def jeffries_xray_report_targ(coord, jeffries_id):
    global _jeffries_xray_report_targ
    try:
        return _jeffries_xray_report_targ[jeffries_id]
    except KeyError:
        pass
    import astroquery.vizier
    v = astroquery.vizier.Vizier(columns=['_r', '*', 'e_Flux', 'Flux', 'logLX',
                                          'ACISBr', 'ACISVar', 'HRCBr', 'HRCVar',
                                          'e_FX', 'FX', 'e_CRate', 'CRate',
                                          '[DFM2003]', '[PMD2006]', 'CR', 'e_CR',
                                          'Signi'])

    x = v.query_region(coord, radius='2s',
                       catalog=['J/A+A/456/977/table2', 'J/A+A/450/993/tableb1',
                                'J/ApJ/588/1009', 'J/ApJ/606/466', 'J/A+A/312/818'])

    ret = []
    if u'J/A+A/456/977/table2' in x.keys():
        recs = x[u'J/A+A/456/977/table2']
        for rec in recs:
            if rec['JTH'] != jeffries_id:
                continue
            fstr = 'M06 reports a count rate of ${:.2f}\\pm{:.2f}$ ct/s in the 0.2--2~keV band.'
            ret.append(fstr.format(rec['CRate'], rec['e_CRate']))

    if u'J/ApJ/588/1009/table4' in x.keys():
        recs = x[u'J/ApJ/588/1009/table4']
        for rec in recs:
            if rec['JTH'] != jeffries_id:
                continue
            fstr = ('D03 reports a flux of ${:.2f}\\pm{:.2f}\\ '
                    '\\mathrm{{ 10^{{-6}} ct s^{{-1}} cm^{{-2}}}}$.')
            ret.append(fstr.format(rec['Flux'], rec['e_Flux']))

    if u'J/ApJ/588/1009/table5' in x.keys():
        recs = x[u'J/ApJ/588/1009/table5']
        for rec in recs:
            if rec['JTH'] != jeffries_id:
                continue
            fstr = 'D03 reports $\\log(L_x)= {:.2f}\\ \\mathrm{{erg/s}}$.'
            ret.append(fstr.format(rec['logLX']))

    if u'J/ApJ/588/1009/table6' in x.keys():
        recs = x[u'J/ApJ/588/1009/table6']
        for rec in recs:
            if rec['JTH'] != jeffries_id:
                continue
            fstr = 'D03 reports $\\log(L_x)\\leq{:.2f}\\ \\mathrm{{erg/s}}$.'
            ret.append(fstr.format(rec['logLX']))

    if u'J/ApJ/606/466/table1' in x.keys():
        recs = x[u'J/ApJ/606/466/table1']
        for rec in recs:
            fstr = ('W04 target {} is {:.2f} arcseconds away and is{} reported '
                    'as binary. {} {}')
            acislut = {'QL': 'quiescent level', 'None': 'no',
                       'QL only': 'only quiescent level',
                       'Stochastic': 'stochastic',
                       'Flare': 'flaring'}
            ast = ''
            if rec['ACISBr'] not in ('Off-chip', 'CCD=s3'):
                ast = 'They report it is {} in ACIS exposures'.format(rec['ACISBr'].lower())
            if rec['ACISBr'] not in ('Off-chip', 'Faint', 'CCD=s3'):
                ast += ' with {} variability.'.format(acislut[rec['ACISVar']])
            else:
                ast += '.'  # ' but do not report variability.'

            hst = 'In HRC exposures it is {}'.format(rec['HRCBr'].lower())
            if rec['HRCBr'] is not 'Faint':
                hst += ' with {} variability.'.format(acislut[rec['HRCVar']])
            else:
                hst += '.'  # ' and do not report variability.'

            ret.append(fstr.format(rec['__DFM2003_'], rec['_r'],
                                   ' not' if rec['Bin'] == 'No' else '', ast, hst))

    if u'J/A+A/450/993/tablea1' in x.keys():
        recs = x[u'J/A+A/450/993/tablea1']
        for rec in recs:
            fstr = ('P06 target {} is {:.2f} arcseconds away, a ${:.2f}\\pm{:.2f}$ ct/ks'
                    ' source with a significance of {:.2f}.')
            ret.append(fstr.format(rec['__PMD2006_'], rec['_r'], rec['CR'], rec['e_CR'],
                                   rec['Signi']))

    if u'J/A+A/450/993/tableb1' in x.keys():
        recs = x[u'J/A+A/450/993/tableb1']
        for rec in recs:
            if rec['JTH'] != jeffries_id:
                continue
            fstr = ('P06 reports this J01 star as having an Xray flux of '
                    '${:.1}\\pm${:.1}\\ \\mathrm{{(10^{{-18}}\\ W\\m^{{-2}}}}$ they '
                    'also report X-ray luminosity and bolometric luminosity.')
            ret.append(fstr.format(rec['FX'], rec['e_FX']))

    _jeffries_xray_report_targ[jeffries_id] = ret
    return ret
