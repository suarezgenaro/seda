#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='seda',
      version='1.0',
      description='Spectral Energy Distribution Analyzer: fit atmospheric models to spectra and/or photometry',
      keywords = ['ultracool dwarfs', 'low mass stars', 'brown dwarfs', 'gas planets', \
                  'spectroscopy', 'photometry', 'atmospheres', 'astronomy', 'astrophysics'],
      author='Genaro Suárez',
      author_email='gsuarez2405@gmail.com',
      url='https://github.com/suarezgenaro/seda',
      license='MIT',
      packages=['seda', 'seda.spectral_indices.spectral_indices'],
      install_requires=['astropy','corner','dynesty','lmfit','matplotlib','numpy','scipy','specutils','spectres','tqdm','xarray'], 
      zip_safe=False,
      include_package_data=True)
