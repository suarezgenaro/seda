#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup
import os
import re


with open('README.md', 'r') as f:
    long_description = f.read()

#def install_requires():
#    reqs = []
#    for line in open('requirements.txt', 'r').readlines():
#        reqs.append(line)
#    return reqs

try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

# read code version
dir_path = os.path.dirname(os.path.realpath(__file__))
version_string = open(os.path.join(dir_path, 'seda',
                      '_version.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, version_string, re.M)
if mo:
	__version__ = mo.group(1)
else:
	raise RuntimeError("Unable to find version string in %s." % (version_string,))

setup(name='seda',
      version=__version__,
      description='SEDA: Spectral Energy Distribution Analyzer for forward modeling and empirical analyses of ultracool objects',
      long_description=long_description, 
      keywords = ['ultracool objects', 'low mass stars', 'brown dwarfs', 'gas giant planets', \
                  'spectroscopy', 'photometry', 'atmospheres', 'astronomy', 'astrophysics'],
      author='Genaro Suárez',
      author_email='gsuarez2405@gmail.com',
      url='https://github.com/suarezgenaro/seda',
      license='MIT',
#      packages = find_packages(),
      packages=['seda', 'seda.spectral_indices', 'seda.synthetic_photometry'],
      install_requires=['astropy','corner','dynesty','lmfit','matplotlib','numpy','prettytable','scipy','specutils','spectres','tqdm','xarray'], 
#      install_requires=install_requires(),
      package_dir = {'seda': 'seda'},
      package_data = {'seda': ['models_aux/model_coverage/*', 'models_aux/model_specifics/*', 'models_aux/model_spectra/*']},
      zip_safe=False,
      include_package_data=True)
