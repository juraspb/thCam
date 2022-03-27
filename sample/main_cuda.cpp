#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/cuda.hpp>
#include "opencv2/core/cuda_types.hpp"

using namespace cv;
using namespace std;

int main()                   
{
  Mat image;
  cuda::GpuMat dst, src;
  string filename="/home/yuriy/Videos/ulc6.avi";
  //string filename="/home/yuriy/Videos/num1.rec";

  VideoCapture cap;
  cap.open(filename);

  if(cap.isOpened()) { // check if we succeeded
    int step = 1;
    // Create a window for display.
    namedWindow( "Image", WINDOW_AUTOSIZE ); 
    namedWindow( "Gray", WINDOW_AUTOSIZE );  
    namedWindow( "Canny", WINDOW_AUTOSIZE );
    namedWindow( "Mask", WINDOW_AUTOSIZE );  
    namedWindow( "Contur", WINDOW_AUTOSIZE );
    TickMeter timer;
    do {
      cout << "FrameStep=" << dec << step << endl ; // вывод в десятичном виде

      //double startCount = getTickCount();
      timer.start();

      Mat image;
      do {
        //cap >> image;
        //if( image.empty() ) {    // Check for invalid input
        bool success = cap.read(image);
        if(! success ) {    // Check for invalid input
          cout <<  "End file" << endl ;
          return -2;
        }
      } while (--step>0);


      src.upload(image);
	    //Ptr<cuda::CLAHE> ptr_clahe = cuda::createCLAHE(5.0, Size(8, 8));
      GaussianBlur(src, src, Size(5, 5), 3, 1.5);

      ptr_clahe->apply(src, dst);
	    dst.download(image);

      double execTime = timer.getAvgTimeMilli();
      timer.stop();
      cout << "execTime = " << execTime << endl;

      GaussianBlur(image, image, Size(5, 5), 3, 1.5);

      Mat kernel =  Mat::ones(5, 5, CV_8UC1);
      dilate(image, image, kernel);
      Mat gray_image;
      cvtColor( image, gray_image, COLOR_BGR2GRAY );
      Mat edge;
      Canny( gray_image, edge, 50, 200 );


      //int64 time = getTickCount();
      //float execTime = (time - startCount) / getTickFrequency();
      //cout << "execTime = " << execTime << endl;

      // find the contours
      vector< vector<Point> > contours;
      findContours(edge, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
      Mat mask = Mat::zeros(edge.rows, edge.cols, CV_8UC1);
      // CV_FILLED fills the connected components found
      drawContours(mask, contours, -1, Scalar(255), CV_FILLED);
      Mat crop(image.rows, image.cols, CV_8UC3);
      image.copyTo(crop, mask);
      
      imshow( "Image", image );                // Show our image inside it.
      imshow( "Gray", gray_image );              // Show our image inside it.
      imshow( "Canny", edge);
      imshow("Mask", mask);
	    imshow( "Contur", image );

      char key = waitKey(0);
      string msg = "";
      switch (key) {
        case 'q' : { step=-1; break; }
        case 'a' : { step=10; break; }
        case 's' : { step=100; break; }
        case 'd' : { step=1000; break; }
        default : { step=1; break; }
      }
    } while(cap.isOpened()&&(step>=0)); // check if we succeeded
  }
  else {
    cout <<  "File not found" << endl ;
    return -1;
  }
  return 0;
}