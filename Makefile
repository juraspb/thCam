CC = g++
PROG = build/typhoon
SRC = source/typhoon.cpp
OPENCV = `pkg-config opencv4 --cflags --libs`
INC	:= -I include

LIBS = $(OPENCV)

$(PROG):$(SRC)
	$(CC) $(SRC) $(LIBS) $(INC) -g -pthread -o $(PROG)
	