#!/usr/bin/env python

from distutils.core import setup

setup(name='jbastro',
      version='0.2',
      description='python astro library',
      author='Jeb Bailey',
      author_email='baileyji@umich.edu',
      url='',
      packages=['jbastro','jbastro.lacosmics'],
      scripts=[],
      install_requires=['numpy', 'matplotlib'],
      package_data={'jbastro':['f2n_fonts/*','data/*']}
      )
