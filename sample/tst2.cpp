//
//  SimpleTestGPU.cpp
//  Loads an image, calls a GPU enabled function that uses opencv GPUMat.
//  Eliminates copying from GPU device to CPU host using download for the result, but
//  still requires copying from a pre-existing image (not in a unified memory data
//  structure) using the opencv upload from cv::Mat to cv::cuda::GpuMat.
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

cv::Mat testfunction(cv::Mat& h_original) {
    // receives a CPU/host based image.  It must use upload to get image data
    // that is already in host memory into device memory.  From there, it uses
    // unified memory to avoid the download back to host memory.
    // The result is returned in the CPU/host based h_result with no explicit copy
    // required from d_result because they are in unified memory.
    
    unsigned int width  = h_original.size().width;
    unsigned int height = h_original.size().height;
    unsigned int channels = h_original.channels();
    unsigned int pixels = width*height;
    unsigned int frameByteSize = pixels * channels;
    std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<channels<<" channels"<<std::endl;
    
    // create a device original image data structure and upload (copy) the original image from host to device space
    // the upload (copy) does not look like it can be avoided for an array that already resides in host memory --
    // in order to avoid the upload/copy, whatever generated the data would need to place the array in a unified
    // memory data structure that was created in advance.  There does not appear to be a way to retroactively tag
    // data residing in host memory as a unified data structure.
    cv::cuda::GpuMat d_original(height, width, h_original.type());
    d_original.upload(h_original);

    // Define pointer used to create result GpuMat
    void *resultptr;
    // allocate unified memory space for result image and assign it to the resultptr
    cudaMallocManaged(&resultptr, frameByteSize);
    // create the host data structure reference to the result image
    cv::Mat h_result(height, width, h_original.type(), resultptr);
    // create the device data structure reference to the result image
    cv::cuda::GpuMat d_result(height, width, h_original.type(), resultptr);
    
    // perform a GPU operation of some sort.  Using threshold for simple placeholder
    cv::cuda::threshold(d_original, d_result, 128.0, 255.0, cv::THRESH_BINARY);
    
    // no need to download or copy the result image from device to host - it already
    // resides in h_result.
    return h_result;
}

int main(int argc, char *argv[]) {
    
    cv::namedWindow("original image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("modified image", cv::WINDOW_AUTOSIZE );
    cv::String filename = "./lena.jpg";
    cv::Mat image, newimage;
    image = cv::imread(filename);
    if (image.empty()) {
        cout << "could not open or find the image" << endl;
        return -1;
    }
    newimage = testfunction(image);
    
    cv::imshow("original image", image);
    cv::imshow("modified image", newimage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}
