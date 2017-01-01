all: clean build

build:
	g++ -std=c++11 -O2 -Wall main.cpp lib/*.cpp -o main

clean:
	rm main -f
