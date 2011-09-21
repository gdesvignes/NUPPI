#!/usr/bin/env python

"""
Pulsar Data Processing Software for NRT

"""

from distutils.core import setup, Extension
import os
import sys

srcdir = 'python'
doclines = __doc__.split("\n")

setup(
    name        = 'nuppi'
  , version     = '0.1'
  , packages    = ['nuppi']
  , package_dir = {'nuppi' : srcdir}
  , maintainer = "Gregory Desvignes"
  , license = "http://www.gnu.org/copyleft/gpl.html"
  , platforms = ["any"]
  , description = doclines[0]
  , long_description = "\n".join(doclines[2:])
  )
