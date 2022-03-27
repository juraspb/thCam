#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <thread>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h>

using namespace std;
using namespace cv;
using namespace cuda;

#define NANO 1000000000L

string txt;

void VES249(string msg) {
    struct timespec last = {}, now = {};
    clock_gettime(CLOCK_MONOTONIC_RAW, &last);
    
    for (int i = 0; i<120; i++) {

        int64_t tosleep=0;
	    do {        
	        clock_gettime(CLOCK_MONOTONIC_RAW, &now);
	        tosleep = ((now.tv_sec - last.tv_sec)*NANO + now.tv_nsec - last.tv_nsec)/1000000;
            cout << "tmr=" << tosleep << "\r";
        } while(tosleep<500);
        last = now;
        txt = "Counter=" + to_string(i) + msg + "\n";
        cout << txt;
    }
}


int main(){
    //thread th1(VES249, " threadMSG");
    namedWindow( "CAM1", WINDOW_NORMAL );
    namedWindow( "CAM2", WINDOW_NORMAL );
    resizeWindow( "CAM1", 640, 480);
    resizeWindow( "CAM2", 640, 480);
    moveWindow( "CAM1", 0, 0);
    moveWindow( "CAM2", 700, 0);

    //string cam1_url = "~/Videos/ul1.rec";
    string cam1_url = "rtsp://root:root@192.168.1.99:554/av0_1";
    //string cam1_url = "rtsp://admin:pP@697469@192.168.1.102:554/Stream/Channel/101" ;
    //string cam2_url = "rtsp://user:9999@192.168.1.167:8557/PSIA/Streaming/channels/2?videoCodecType=H.264";
    //string cam2_url = "rtsp://192.168.1.167:8555/PSIA/Streaming/channels/0?videoCodecType=MJPEG";
    //string filename="/home/yuriy/Videos/ul1.rec";


    VideoCapture cam1cap, cam2cap;
    int i=0;
    char key;
    cam1cap.open(cam1_url);
    //cam2cap.open(cam2_url);
    
    //if (cam1cap.isOpened()or cam2cap.isOpened()){
        //Mat cam1img,cam2img;
    if (cam1cap.isOpened()){
        Mat cam1img;
        GpuMat img_gpu,gray_gpu,edged_gpu;
        Ptr<CannyEdgeDetector> cannyFilter = createCannyEdgeDetector(50, 100);
        Ptr<Filter> gaussianFilter = createGaussianFilter(gray_gpu.type(),gray_gpu.type(),Size(7, 7),0);
        
        while (1) {	
          if (cam1cap.read(cam1img)) {
	          imshow("CAM2", cam1img);
            Mat img;
            img_gpu.upload(cam1img);
            auto start = std::chrono::high_resolution_clock::now(); 
            cuda::cvtColor(img_gpu, gray_gpu, COLOR_BGR2GRAY);
            gaussianFilter->apply(gray_gpu, gray_gpu);
            cannyFilter->detect(gray_gpu, edged_gpu);
            auto finish = std::chrono::high_resolution_clock::now(); 
            edged_gpu.download(img);
            std::chrono::duration<double> elapsed_time = finish - start;
            std::cout << "Execute Time: " << elapsed_time.count() * 1000 << " msecs" << "\n" << std::endl;
            putText(img, txt, Point(40,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, LINE_AA);
	          imshow("CAM1", img);
          }
/*          
          if (cam2cap.read(cam2img)) {
            Mat img,gray; 
            //putText(cap2img, txt, Point(40,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, LINE_AA); 
            cvtColor(cam2img, gray, COLOR_BGR2GRAY);
            Canny(gray,img,100,200);
	          imshow("CAM2", img);
          }

            //Sobel(cam1img,img,CV_32F,1,1);
            //cvtColor(cam1img, ggray, COLOR_BGR2GRAY);
            //Canny(ggray,img,100,200);


*/          
     	  key = waitKey(1);
          if (key=='q') break;
          if (key=='s') {
            stringstream ss;
            i++ ;
            ss<<"cam1("<<i<<").jpg";
            imwrite(ss.str(),cam1img);
            cout<<"i:"<<i<<" img path:"<<ss.str()<<endl;
          }
/*
          if (key=='d') {
            stringstream ss;
            i++ ;
            ss<<"cam2("<<i<<").jpg";
            //imwrite(ss.str(),cam2img);
            cout<<"i:"<<i<<" img path:"<<ss.str()<<endl;
          }
*/          
        }
    }
    //th1.join();
    return 1 ;
}