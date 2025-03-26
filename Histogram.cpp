#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"


using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		std::cout << "Image Size: " << image_input.size() << " bytes" << std::endl;
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
		
		//Histogram
		int numBins = 256;

		const int histogramSize = numBins*sizeof(int);
		std::vector<int> Hist(numBins);
		
		
		cl::Buffer dev_grey_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer partial_hist(context, CL_MEM_READ_WRITE, histogramSize);
		queue.enqueueWriteBuffer(dev_grey_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		
		cl::Kernel histKernel = cl::Kernel(program, "histogram");
		histKernel.setArg(0, dev_grey_input);
		histKernel.setArg(1, (int) image_input.size());
		histKernel.setArg(2, partial_hist);
		histKernel.setArg(3, numBins);

		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		int preferredWorkSize = histKernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
		//cerr << histKernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (device) << endl;
		int globalWorkSize = ((image_input.size() * preferredWorkSize-1)/preferredWorkSize)*preferredWorkSize;
		
		queue.enqueueNDRangeKernel(histKernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(preferredWorkSize));
		cl_ulong partial_hist_size;
		partial_hist.getInfo(CL_MEM_SIZE, &partial_hist_size);
		std::cout << "Buffer size: " << partial_hist_size << " bytes" << std::endl;

		queue.enqueueReadBuffer(partial_hist, CL_TRUE, 0, histogramSize, &Hist[0]);

		cl::Kernel scanKernel = cl::Kernel(program, "scanBL");
		scanKernel.setArg(0, partial_hist);

		queue.enqueueNDRangeKernel(scanKernel, cl::NullRange, cl::NDRange(numBins), cl::NullRange);
		queue.enqueueReadBuffer(partial_hist, CL_TRUE, 0, histogramSize, &Hist[0]);

		std::vector<float> FloatHist(numBins);
		cl::Kernel normaliseKernel = cl::Kernel(program, "normalise");
		normaliseKernel.setArg(0, partial_hist);
		normaliseKernel.setArg(1, histogramSize);

		queue.enqueueNDRangeKernel(normaliseKernel, cl::NullRange, cl::NDRange(numBins), cl::NullRange);
		queue.enqueueReadBuffer(partial_hist, CL_TRUE, 0, histogramSize, &FloatHist[0]);

		cl::Buffer scaledBuffer(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Kernel scaledKernel = cl::Kernel(program, "scaled");
		scaledKernel.setArg(0, partial_hist);
		scaledKernel.setArg(1, scaledBuffer);
		scaledKernel.setArg(2, numBins);

		queue.enqueueNDRangeKernel(scaledKernel, cl::NullRange, cl::NDRange(numBins), cl::NullRange);
		queue.enqueueReadBuffer(scaledBuffer, CL_TRUE, 0, histogramSize, &Hist[0]);
		std::cout << "Scaled Hist = " << Hist << std::endl;

		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());

		cl::Kernel backProjGrey = cl::Kernel(program, "backProjection");
		backProjGrey.setArg(0, dev_grey_input);
		backProjGrey.setArg(1, dev_image_output);
		backProjGrey.setArg(2, scaledBuffer);

		queue.enqueueNDRangeKernel(backProjGrey, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		vector<unsigned char> output_buffer(image_input.size());

		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, image_input.size(), &output_buffer.data()[0]);
		
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
