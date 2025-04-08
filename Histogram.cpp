#include <iostream>
#include <vector>
#include <cmath>

#include "Utils.h"
#include "CImg.h"


using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
	std::cerr << "  -b : define number of bins (default: 256)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	int numBins = 256;
	int maxValue;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { numBins = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
        CImg<unsigned char> grey_image;
		
		std::cout << "Image Size: " << image_input.size() << " bytes" << std::endl;
		int channels = image_input.spectrum();
        string ColourSpace;
        int preferredWorkSize;
        int globalWorkSize;
		CImgDisplay disp_input(image_input,"input");

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		std::cout << channels << std::endl;
		
        if (channels == 4) {
            //RGBA
            ColourSpace = "RGBA";
			std::cout << ColourSpace << std::endl;
        }
        else if (channels == 3) {
            //RGB
            cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
            cl::Buffer dev_image_grey(context, CL_MEM_READ_WRITE, image_input.size()/channels);

            queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

            cl::Kernel RGBKernel = cl::Kernel(program, "rgb2grey");
            RGBKernel.setArg(0, dev_image_input);
            RGBKernel.setArg(1, dev_image_grey);
			RGBKernel.setArg(2, channels);

            cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
            preferredWorkSize = RGBKernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
            globalWorkSize = ((image_input.size() * preferredWorkSize-1)/preferredWorkSize)*preferredWorkSize;

			queue.enqueueNDRangeKernel(RGBKernel, cl::NullRange, cl::NDRange(image_input.size()/channels), cl::NullRange);

            vector<unsigned char> grey_buffer(image_input.size()/channels);

            queue.enqueueReadBuffer(dev_image_grey, CL_TRUE, 0, grey_buffer.size(), &grey_buffer.data()[0]);

            CImg<unsigned char> temp_grey_image(grey_buffer.data(), image_input.width(), image_input.height());
            ColourSpace = "RGB";
			grey_image.assign(temp_grey_image);
			std::cout << ColourSpace << std::endl;
        }
        else {
            //Greyscale
            grey_image.assign(image_input);
            ColourSpace = "Grey";
			std::cout << ColourSpace << std::endl;
        }

		CImgDisplay disp_grey(grey_image,"greyscale");
		std::cout << (int)grey_image.max() << std::endl;

		for (int i=1; i<17; i++) {
			if (int(pow(2.0, i)) > (int)grey_image.max()) {
				maxValue = pow(2.0, i);
				break;
			}
		}
		std::cout << maxValue << std::endl;

		//Histogram
		std::cout << numBins << std::endl;

		int histogramSize = numBins*4;
		std::vector<int> Hist(numBins);
		
		cl::Buffer dev_grey_input(context, CL_MEM_READ_ONLY, grey_image.size());
		cl::Buffer partial_hist(context, CL_MEM_READ_WRITE, histogramSize);
		queue.enqueueWriteBuffer(dev_grey_input, CL_TRUE, 0, (int)grey_image.size(), &grey_image.data()[0]);
		std::vector<int> localHistogram;
		
		cl::Kernel histKernel = cl::Kernel(program, "histogram");
		histKernel.setArg(0, dev_grey_input);
		histKernel.setArg(1, (int) grey_image.size());
		histKernel.setArg(2, partial_hist);
		histKernel.setArg(3, numBins);
		histKernel.setArg(4, maxValue);
		histKernel.setArg(5, numBins*sizeof(int), NULL);

		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        preferredWorkSize = histKernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
        globalWorkSize = ((image_input.size() * preferredWorkSize-1)/preferredWorkSize)*preferredWorkSize;
		std::cout << preferredWorkSize << std::endl;
		std::cout << globalWorkSize << std::endl;
		
		queue.enqueueNDRangeKernel(histKernel, cl::NullRange, cl::NDRange(numBins), cl::NullRange);
		cl_ulong partial_hist_size;
		partial_hist.getInfo(CL_MEM_SIZE, &partial_hist_size);
		std::cout << "Buffer size: " << partial_hist_size << " bytes" << std::endl;

		queue.enqueueReadBuffer(partial_hist, CL_TRUE, 0, histogramSize, &Hist[0]);
		std::cout << "Original Hist = " << Hist << std::endl;

		cl::Kernel scanKernel = cl::Kernel(program, "scanBL");
		scanKernel.setArg(0, partial_hist);

		queue.enqueueNDRangeKernel(scanKernel, cl::NullRange, cl::NDRange(numBins), cl::NullRange);
		queue.enqueueReadBuffer(partial_hist, CL_TRUE, 0, histogramSize, &Hist[0]);
		std::cout << "Scan Hist = " << Hist << std::endl;

		std::vector<float> FloatHist(numBins);
		cl::Kernel normaliseKernel = cl::Kernel(program, "normalise");
		normaliseKernel.setArg(0, partial_hist);
		normaliseKernel.setArg(1, histogramSize);

		queue.enqueueNDRangeKernel(normaliseKernel, cl::NullRange, cl::NDRange(numBins), cl::NullRange);
		queue.enqueueReadBuffer(partial_hist, CL_TRUE, 0, histogramSize, &FloatHist[0]);
		std::cout << "Float Hist = " << FloatHist << std::endl;

		cl::Buffer scaledBuffer(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Kernel scaledKernel = cl::Kernel(program, "scaled");
		scaledKernel.setArg(0, partial_hist);
		scaledKernel.setArg(1, scaledBuffer);
		scaledKernel.setArg(2, numBins);
		scaledKernel.setArg(3, maxValue);

		queue.enqueueNDRangeKernel(scaledKernel, cl::NullRange, cl::NDRange(numBins), cl::NullRange);
		queue.enqueueReadBuffer(scaledBuffer, CL_TRUE, 0, histogramSize, &Hist[0]);
		std::cout << "Scaled Hist = " << Hist << Hist.size() << std::endl;

		int scaleFactor = 256/numBins;

		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		std::vector<unsigned char> output_buffer(image_input.size()/channels);
        if (ColourSpace == "Grey") { //differing projections for greyscale, rgb, rgba
			cl::Kernel backProjGrey = cl::Kernel(program, "backProjection");
			backProjGrey.setArg(0, dev_grey_input);
			backProjGrey.setArg(1, dev_image_output);
			backProjGrey.setArg(2, scaledBuffer);
			backProjGrey.setArg(3, scaleFactor);
			
			queue.enqueueNDRangeKernel(backProjGrey, cl::NullRange, cl::NDRange(grey_image.size()), cl::NullRange);

			std::cout << "Here1" << std::endl;

			cl_ulong dev_image_size;
			dev_image_output.getInfo(CL_MEM_SIZE, &dev_image_size);
			std::cout << "Image size:  " << dev_image_size << " bytes" << std::endl;
			std::cout << "Buffer size: " << output_buffer.size() << " bytes" << std::endl;

			queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, image_input.size(), &output_buffer.data()[0]);
		}
		else {
			cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
			queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

			cl::Kernel backProjColour = cl::Kernel(program, "backProjRGBA");
			backProjColour.setArg(0, dev_image_input);
			backProjColour.setArg(1, dev_image_output);
			backProjColour.setArg(2, scaledBuffer);
			backProjColour.setArg(3, maxValue);
			backProjColour.setArg(4, channels);
			backProjColour.setArg(5, scaleFactor);
			
			queue.enqueueNDRangeKernel(backProjColour, cl::NullRange, cl::NDRange(grey_image.size()), cl::NullRange);

			std::cout << "Here1" << std::endl;

			cl_ulong dev_image_size;
			dev_image_output.getInfo(CL_MEM_SIZE, &dev_image_size);
			std::cout << "Image size:  " << dev_image_size << " bytes" << std::endl;
			std::cout << "Buffer size: " << output_buffer.size() << " bytes" << std::endl;

			queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, image_input.size(), &output_buffer.data()[0]);
		}
		std::cout << "Here2" << std::endl;
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), 1);
		CImgDisplay disp_output(output_image,"output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
