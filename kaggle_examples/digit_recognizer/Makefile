all: clean build

build: learn predict

learn:
	rm -f learn
	g++ -std=c++11 -O2 -Wall learn.cpp ../../lib/*.cpp -I../../ -o learn

predict:
	rm -f predict
	g++ -std=c++11 -O2 -Wall predict.cpp ../../lib/*.cpp -I../../ -o predict

clean:
	rm learn predict -f
