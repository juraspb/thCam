//
//  SimpleTestGPU.cpp
//  Loads an image, calls a GPU enabled function that uses opencv GPUMat.
//  Eliminates copying from GPU device to CPU host using download for the result, but
//  still requires copying from a pre-existing image 
//

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h> 

// cuda stuff
#include <opencv2/cudaarithm.hpp>
// Nvidia cuda api
#include <cuda_runtime.h>

using namespace std;

void OpencvCudaProcess(const cv::cuda::GpuMat& d_original, cv::cuda::GpuMat& d_result) {
	// perform a GPU operation of some sort.  Using threshold for simple placeholder
	cv::cuda::threshold(d_original, d_result, 128.0, 255.0, cv::THRESH_BINARY);
}

int main(int argc, char *argv[]) {

	cv::namedWindow("original image", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("modified image", cv::WINDOW_AUTOSIZE );
	cv::String filename = "/usr/share/visionworks/sources/data/lena.jpg";

	// Max size buffer. Be aware of your max number of channels and format size (UC, F32, F64) for computing max byte size 
	const unsigned int maxImageByteSize = 1920*1080*3;

	void *image_in, *image_out;
    	// allocate unified memory space for INPUT image 
    	if (cudaSuccess != cudaMallocManaged(&image_in, maxImageByteSize))
		return (-1);
     	// allocate unified memory space for OUTPUT image 
   	if (cudaSuccess != cudaMallocManaged(&image_out, maxImageByteSize))
		return (-2);

        /* Process image, this block may be in a loop */
	{
		cv::Mat read_img = cv::imread(filename);
		if (read_img.empty()) {
		    cout << "could not open or find the image" << endl;
		    return (-3);
		}

		/* Prepare appropriate Mats (headers for size and format, specifying our unified memory buffers */
 		cv::Mat image(read_img.rows, read_img.cols, read_img.type(), image_in);
		cv::Mat newimage(read_img.rows, read_img.cols, read_img.type(), image_out);

		/* Prepare appropriate GpuMats (headers for size and format, specifying our unified memory buffers */ 
		cv::cuda::GpuMat d_image(read_img.rows, read_img.cols, read_img.type(), image_in);
		cv::cuda::GpuMat d_newimage(read_img.rows, read_img.cols, read_img.type(), image_out);

// Both options should work, you would check timings for your case
//#define COPY_TO_CPU
#ifdef COPY_TO_CPU
		read_img.copyTo(image);
		if (image.data != image_in) {
			std::cerr << "Error: image data buffer changed from initial unified memory" << std::endl;
			return (-4);
		}
#else /* COPY_TO_GPU */
		d_image.upload(read_img);
		if (d_image.data != image_in) {
			std::cerr << "Error: d_image data buffer changed from initial unified memory" << std::endl;
			return (-5);
		}	
#endif

		/* Process image on GPU */
		OpencvCudaProcess(d_image, d_newimage);

		cv::imshow("original image", image);

		/* Display result from CPU Mat without copy */
		cv::imshow("modified image", newimage);

		/* Wait for key press */
		cv::waitKey(-1);
	}

	cudaFree(image_in);
	cudaFree(image_out);
	cv::destroyAllWindows();

	return 0;
}
