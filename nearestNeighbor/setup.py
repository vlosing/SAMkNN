 
from distutils.core import setup, Extension
import numpy as np
from distutils.sysconfig import get_python_inc
nnExtension = Extension('libNearestNeighbor',
                    include_dirs = [get_python_inc(), np.get_include()],
                    libraries = [],
                    library_dirs = [],
		    extra_compile_args = ['-O3'],
                    sources = ['nearestNeighbor.cpp'])

setup (name = 'nearestNeighbor',
       version = '1.0',
       description = 'Calculation for nearest neighbor',
       author = 'Viktor Losing',
       author_email = 'vlosing@techfak.uni-bielefeld.de',
       install_requires=["numpy"],
       ext_modules = [nnExtension])

