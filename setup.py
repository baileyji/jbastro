#!/usr/bin/env python

from distutils.core import setup

pyastro_dep_link=('https://github.com/sczesla/PyAstronomy/tarball/'
                  'v_0-7-0#egg=PyAstronomy-0.7.0')

setup(name='jbastro',
      version='0.5',
      description='python astro library',
      author='Jeb Bailey',
      author_email='baileyji@ucsb.edu',
      url='',
      packages=['jbastro','jbastro.lacosmics'],
      scripts=[],
      install_requires=['numpy', 'matplotlib',
                        'PyEphem','mechanize','BeautifulSoup',
                        'PyAstronomy'],
      package_data={'jbastro':['f2n_fonts/*','data/*']},
      dependency_links = [pyastro_dep_link]
      )
      
