DATE = $(shell date +%d%b%y)

install:
	make slalib
	python setup.py install
	cd src; make clean; make

clean:
	cd src; make clean; cd ..; rm -f slalib/*.o slalib/sla_test;
package:
	make clean
	cd ..; tar cvfz nuppi_$(DATE).tar.gz nuppi --exclude .git 


# SLALIB STUFF

FC = gfortran
FFLAGS = -g -O2 -fPIC

slalib: libsla.so
	cd slalib ; $(FC) -o sla_test sla_test.f -fno-second-underscore -lsla
	slalib/sla_test

libsla.so:
	cd slalib ; $(FC) $(FFLAGS) -fno-second-underscore -c -I. *.f *.F
	rm slalib/sla_test.o
	cd slalib ; $(FC) -shared -o /usr/lib/libsla.so -fno-second-underscore *.o
