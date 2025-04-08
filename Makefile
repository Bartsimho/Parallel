assessment: Histogram.cpp
	g++ -std=c++0x RGB.cpp -o RGB -lOpenCL -lX11 -lpthread
	g++ -std=c++0x Histogram.cpp -o Histogram -lOpenCL -lX11 -lpthread
clean:
	rm Histogram
	rm RGB