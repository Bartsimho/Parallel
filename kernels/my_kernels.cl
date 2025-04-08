
//rgb2gray
kernel void rgb2grey(global const uchar* A, global uchar* B, int channels) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/channels; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	
	float R = A[id + 0];  // Red component
    float G = A[id + 1];  // Green component
    float Bl = A[id + 2];  // Blue component
        
    // Calculate luminance using the formula
    int Y = 0.2126f * R + 0.7152f * G + 0.0722f * Bl;

    // Store the result in the output array
    B[id] = Y;
}

kernel void histogram( __global uchar* data , int numData , global int* histogram, int numBins, int maxValue, __local int* localHistogram) {
	//__local int localHistogram [256];
	int lid = get_local_id(0);
	int gid = get_global_id(0);
	for ( int i = lid ; i < numBins; i += get_local_size(0))
	{
		localHistogram[i] = 0;
	}
	barrier (CLK_LOCAL_MEM_FENCE);

	for ( int i = gid; i < numData; i += get_global_size(0))
	{
		if(gid < numData) {
			int binIndex = (data[gid]*numBins)/maxValue;
			atomic_add(&localHistogram[binIndex], 1);
		}
	}
	barrier (CLK_LOCAL_MEM_FENCE);
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

kernel void scaled(global float* histogram, global int* scaledHistogram, const int numBins, const int maxValue) {
	int gid = get_global_id(0);
	if (gid < numBins) {
		scaledHistogram[gid] = (int) (histogram[gid]*(maxValue-1.0f));
	}
}

kernel void backProjection(global const uchar* greyImage, global uchar* backProjImage, global const int* scaledHistogram, int scaleFactor) {
    int gid = get_global_id(0);

    if (gid == 0) {
        printf("First pixel intensity: %d\n", greyImage[gid]);
        printf("Scaled value: %d\n", scaledHistogram[greyImage[gid]]);
    }
	int intensity = greyImage[gid]/scaleFactor;
    backProjImage[gid] = scaledHistogram[intensity];

}

kernel void backProjRGBA(global const uchar* colourImage, global uchar* backProjImage, global const int* scaledHistogram, const int maxValue, const int channels, int binScale) {
    int gid = get_global_id(0);  // Pixel index
    int image_size = get_global_size(0)/channels;

    float R = colourImage[gid + 0];  // Red component
    float G = colourImage[gid + 1];  // Green component
    float Bl = colourImage[gid + 2];  // Blue component

    int intensity = (0.2126f * R + 0.7152f * G + 0.0722f * Bl)/binScale;

    float scaleFactor = scaledHistogram[intensity]; // Normalize LUT value to [0,1]

    backProjImage[gid] = (R * scaleFactor);
    backProjImage[gid + 1] = (G * scaleFactor);
    backProjImage[gid + 2] = (Bl * scaleFactor);
	if (channels == 4) {
		int A = colourImage[gid + 3];
		backProjImage[gid + 3] = A;
	}
}