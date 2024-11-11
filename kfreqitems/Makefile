# ------------------------------------------------------------------------------
#  Makefile 
# ------------------------------------------------------------------------------
ALLOBJS = util.o lsh.o bucket.o bin.o assign.o eval.o seeding.o main.o

OPENMP  = -Xcompiler -fopenmp
OPT     = -w -O3

# For 172.28.177.86
# ARCH    = -gencode arch=compute_61,code=sm_61
# CFLAGS  = -I/usr/mpi/gcc/openmpi-1.10.4-cuda-hfi/include
# LDFLAGS = -L/usr/mpi/gcc/openmpi-1.10.4-cuda-hfi/lib64

# For iGyro
ARCH	=	-gencode arch=compute_90,code=sm_90
CFLAGS	=	-I/usr/lib/x86_64-linux-gnu/openmpi/include
LDFLAGS	=	-L/usr/src/linux-headers-5.4.0-187/include/linux

FLAGS   = 	-lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm

# ------------------------------------------------------------------------------
#  Compiler with OpenMP (first .cu, then .cc, order-sensitive)
# ------------------------------------------------------------------------------
all:$(ALLOBJS)
	nvcc -std=c++11 $(OPENMP) $(OPT) -o silk $(ARCH) -lm -lcudart -lcublas \
		$(CFLAGS) $(LDFLAGS) $(FLAGS) $(ALLOBJS)

%.o: %.cu
	nvcc -std=c++11 $(OPENMP) -c $(OPT) $(ARCH) $(CFLAGS) -o $@ $<

%.o: %.cc
	nvcc -std=c++11 $(OPENMP) -c $(OPT) $(ARCH) $(CFLAGS) -o $@ $<

clean:
	-rm $(ALLOBJS) silk
