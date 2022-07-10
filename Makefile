PETSC_DIR=/media/dirkmunro/Terra/Code/petsc
PETSC_ARCH=arch-linux-c-debug
CFLAGS = -I.
FFLAGS=
CPPFLAGS=-I-
#/usr/include/python3.8/ -lpython3.8
FPPFLAGS=
LOCDIR=
EXAMPLESC=
EXAMPLESF=
MANSEC=
CLEANFILES=
NP=


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

main: main.o
	$(shell ./make.sh)

topopt: main.o TopOpt.o LinearElasticity.o MMA.o Filter.o PDEFilter.o MPIIO.o chkopts
	rm -rf topopt
	-${CLINKER} -o topopt main.o TopOpt.o LinearElasticity.o MMA.o Filter.o PDEFilter.o MPIIO.o ${PETSC_SYS_LIB} -I/usr/include/python3.8/ -lpython3.8
	${RM}  main.o TopOpt.o LinearElasticity.o MMA.o Filter.o PDEFilter.o MPIIO.o
	rm -rf *.o

myclean:
	rm -rf topopt *.o output* binary* log* makevtu.pyc Restart* 

