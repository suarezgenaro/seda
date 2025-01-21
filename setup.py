#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

#with open("README", 'r') as f:
#    long_description = f.read()

#def install_requires():
#    reqs = []
#    for line in open('requirements.txt', 'r').readlines():
#        reqs.append(line)
#    return reqs

setup(name='seda',
      version='1.0',
      description='SEDA: Spectral Energy Distribution Analyzer',
#      long_description=long_description, 
      keywords = ['ultracool objects', 'low mass stars', 'brown dwarfs', 'gas giant planets', \
                  'spectroscopy', 'photometry', 'atmospheres', 'astronomy', 'astrophysics'],
      author='Genaro Suárez',
      author_email='gsuarez2405@gmail.com',
      url='https://github.com/suarezgenaro/seda',
      license='MIT',
      packages=['seda', 'seda.spectral_indices', 'seda.synthetic_photometry'],
      install_requires=['astropy','corner','dynesty','lmfit','matplotlib','numpy','scipy','specutils','spectres','tqdm','xarray'], 
#      install_requires=install_requires(),
      zip_safe=False,
      include_package_data=True)
