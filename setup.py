#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='seda',
      version='1.0',
      description='Spectral Energy Distribution Analyzer: fit atmospheric models to spectra and/or photometry',
      keywords = ['ultracool dwarfs', 'low mass stars', 'brown dwarfs', 'gas planets', \
                  'spectroscopy', 'photometry', 'atmospheres', 'astronomy', 'astrophysics'],
      author='Genaro Suarez',
      author_email='gsuarez2405@gmail.com',
      url='https://github.com/suarezgenaro/seda',
      license='MIT',
      packages=['seda'],
      install_requires=['numpy','astropy','matplotlib','scipy'], # complete the list
      zip_safe=False,
      include_package_data=True)
