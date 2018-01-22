rm cylib.so

cython -a cylib.pyx -o cylib.cc

#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
#  -I/usr/include/python3.5m -o cylib.so cylib.c
g++ -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
  -I/usr/include/python3.6m -o cylib.so cylib.cc
#g++ -shared -pthread -fopenmp -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
#  -I/usr/include/python3.5m -o cylib.so cylib.cc
