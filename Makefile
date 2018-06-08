BINS = bin/kinfu bin/surfel
CXX = g++
NVCC = nvcc

INCLUDES = -I/usr/local/cuda/include -I/usr/include/eigen3 -Iinclude
DEFINES =
CXXFLAGS = -O2 -Wall -std=c++11 -ffast-math $(INCLUDES) $(DEFINES)
NVCCFLAGS = -O2 -std=c++11 -arch=sm_35 -lineinfo $(INCLUDES) $(DEFINES)
LDFLAGS = -L/usr/local/cuda/lib64
LDLIBS = -lpng -lm -lglfw -lGL -lcudart -lOpenNI2

SOURCES := $(shell find src -name *.cpp -or -name *.cu)
OBJECTS := $(SOURCES:%=bin/%.o)

all: $(BINS)

bin/kinfu: apps/kinfu/main.cpp $(OBJECTS)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

bin/surfel: apps/surfel/main.cpp $(OBJECTS)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

bin/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

bin/%.cu.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc -c -o $@ $<

.PHONY: clean

clean:
	$(RM) -rf $(OBJECTS) $(BINS)
