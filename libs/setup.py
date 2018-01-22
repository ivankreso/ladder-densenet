from distutils.core import setup
from Cython.Build import cythonize

setup(
  ext_modules = cythonize("cylib.pyx")
)

#setup(ext_modules = cythonize(
#           "clib.pyx",                 # our Cython source
#           #sources=["Rectangle.cpp"],  # additional source file(s)
#           language="c++",             # generate C++ code
#))
