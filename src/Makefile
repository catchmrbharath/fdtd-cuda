CC = nvcc
OBJECTS = fdtd.o drude_mode.o tm_mode.o kernels.o pml_mode.o
FINAL = fdtd
LDFLAGS = -lglut -lGL -lpthread


all : $(FINAL)

$(FINAL) : $(OBJECTS)
	nvcc -g -G $(LDFLAGS)  $(OBJECTS) -o  $(FINAL)

.cu.o:
	$(CC) -g -G -c $<

.SUFFIXES: .cu
#drude_mode.o: drude_mode.cu
#	nvcc -c drude_mode.cu
#
#tm_mode.o: tm_mode.cu
#	nvcc -c tm_mode.cu
#
#kernels.o: kernels.cu
#	nvcc -c kernels.cu
#
#pml_mode.o: pml_mode.cu
#	nvcc -c pml_mode.cu

clean:
	rm -rf *.o $(FINAL)
