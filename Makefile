CC = g++
PROG = build/typhoon
SRC = source/typhoon.cpp
OPENCV = `pkg-config opencv4 --cflags --libs`

LIBS = $(OPENCV)

$(PROG):$(SRC)
	$(CC) $(SRC) -g -pthread -o $(PROG) $(LIBS)
	