CXX = g++-8
# opencv
LIBS = $(shell pkg-config --libs opencv)
CFLAGS = $(shell pkg-config --cflags opencv)
# tensorlow
TF_ROOT = libs/libtensorflow/
LIBS += -Wl,-rpath,$(TF_ROOT)/lib -L$(TF_ROOT)/lib -ltensorflow
CFLAGS += -I$(TF_ROOT)/include
# c flags
CFLAGS += -std=c++11 -Wall -O3 -msse2 -lm -Wno-unused-result -Wno-sign-compare -Wno-unused-variable -Wno-parentheses -Wno-format

BIN = ./bin/detect_face
.PHONY: clean all

all: ./bin $(BIN)

./bin/detect_face: ./src/detect_face.cpp ./src/utils/*.h

./bin:
	mkdir -p bin

$(BIN) :
	$(CXX) -o $@ $(filter %.cpp %.o %.c, $^) $(CFLAGS) $(LIBS) 
$(OBJ) :
	$(CXX) -o $@ $(firstword $(filter %.cpp %.c, $^) ) -c $(CFLAGS) $(LIBS)

clean :
	rm -rf bin