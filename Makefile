CXX = g++

# opencv
LIBS = $(shell pkg-config --libs opencv)
CFLAGS = $(shell pkg-config --cflags opencv)


# tensorflow
TF_ROOT = libs/libtensorflow/
LIBS += -Wl,-rpath,$(TF_ROOT)/lib -L$(TF_ROOT)/lib -ltensorflow
CFLAGS += -I$(TF_ROOT)/include

# c flags
CFLAGS += -std=c++11 -Wall -O3 
LDFLAGS = -lm -Wno-unused-result -Wno-sign-compare -Wno-unused-variable -Wno-parentheses -Wno-format -Wno-unused-command-line-argument -Wno-unused-function


BIN = ./bin/detect_face ./bin/camera_detect
OBJ = ./bin/tensorflow_mtcnn.o ./bin/utils_mtcnn.o ./bin/mtcnn.o

.PHONY: clean all

all: bin $(BIN) 

$(BIN) : $(OBJ) ./src/utils/*.hpp

$(BIN) : %:%.o


%:%.o
	@echo "Compling $@"
	@$(CXX) $< -o $@ $(OBJ) $(CFLAGS) $(LIBS) $(LDFLAGS)

bin/%.o : src/%.cpp 
	@echo "Compling $@"
	@$(CXX) $(CFLAGS) $(LIBS) $(LDFLAGS) -c $< -o $@ 

bin/%.o : src/mtcnn/%.cpp src/mtcnn/%.hpp
	@echo "Compling $@"
	@$(CXX) $(CFLAGS) $(LIBS) $(LDFLAGS) -c $< -o $@

bin:
	mkdir -p bin

clean :
	rm -rf bin