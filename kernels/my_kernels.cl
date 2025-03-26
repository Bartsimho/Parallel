//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	if (colour_channel == 0) {
		B[id] = A[id];
	}
	else {
		B[id] = 0;
	}
}

//inversion
kernel void invert(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = 255 - A[id];
}

//rgb2gray
kernel void rgb2gray(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	
	float R = A[id + 0];  // Red component
        float G = A[id + 1];  // Green component
        float Bl = A[id + 2];  // Blue component
        
        // Calculate luminance using the formula
        int Y = 0.2126f * R + 0.7152f * G + 0.0722f * Bl;

        // Store the result in the output array
        B[id] = Y;
}

kernel void histogram( __global uchar* data , int numData , global int* histogram, int numBins) {
	__local int localHistogram [256] ;
	int lid = get_local_id(0) ;
	int gid = get_global_id(0) ;
	for ( int i = lid ; i < numBins; i += get_local_size(0))
	{
		localHistogram[i] = 0;
	}
	barrier (CLK_LOCAL_MEM_FENCE) ;

	for ( int i = gid ; i < numData ; i += get_global_size(0))
	{
		if(gid < numData) {
			int binIndex = (data[gid]*numBins)/256;
			atomic_add(&localHistogram[binIndex], 1);
		}
	}
	barrier (CLK_LOCAL_MEM_FENCE) ;
	for ( int i = lid ; i < numBins; i += get_local_size(0))
	{
		atomic_add(&histogram[i], localHistogram[i]);
	}
}

kernel void scanBL(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

kernel void normalise(global float* histogram, const uint NumberOfPixels){
	int gid = get_global_id(0);
	int N = get_global_size(0);

	if (gid < NumberOfPixels) {
		histogram[gid] = histogram[gid]/histogram[N-1];
	}
}

kernel void scaled(global float* histogram, global int* scaledHistogram, const int numBins) {
	int gid = get_global_id(0);

	if (gid < numBins) {
		scaledHistogram[gid] = (int) (histogram[gid]*255.0f);
	}
}

kernel void backProjection(global const uchar* grayImage, global uchar* backProjImage, global const uchar* scaledHistogram) {
    int gid = get_global_id(0);

    backProjImage[gid] = scaledHistogram[grayImage[gid]];

}

kernel void backProjColour(global const uchar* colourImage, global uchar* backProjImage, global const uchar* scaledHistogram) {
    int gid = get_global_id(0);  // Pixel index
    int image_size = get_global_size(0)/3;

    uchar R = colourImage[gid];
    uchar G = colourImage[gid + image_size];
    uchar B = colourImage[gid + 2 * image_size];

    uchar intensity = (uchar)(0.2126f * R + 0.7152f * G + 0.0722f * B);

    float scaleFactor = scaledHistogram[intensity] / 255.0f; // Normalize LUT value to [0,1]

    backProjImage[gid] = (uchar)(R * scaleFactor);
    backProjImage[gid + image_size] = (uchar)(G * scaleFactor);
    backProjImage[gid + 2 * image_size] = (uchar)(B * scaleFactor);
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++) 
		result += A[i + j*width + c*image_size];

	result /= 9;

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++) 
		result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];

	B[id] = (uchar)result;
}
