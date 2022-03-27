#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudabgsegm.hpp>
//#include <opencv2/cudaobjdetect.hpp>
//#include <opencv2/cudaoptflow.hpp>
//#include "opencv2/cudaarithm.hpp"
//#include <omp.h>
//#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

mutex m_console;

void camExecute(string URL)
{
    namedWindow(URL, WINDOW_AUTOSIZE );
    resizeWindow(URL, 640, 480);
    moveWindow(URL, 0, 0);
    VideoCapture cap;
    cap.open(URL);
   for( char i = 0; i < 1000000; i++)
   {
       Mat image,gray_image;
       cap >> image;
       cvtColor( image, gray_image, COLOR_BGR2GRAY );
       imshow(URL, gray_image);
	}	
/*
    for( int i = 0; i < 127; i++){
        lock_guard<mutex> lock(m_console);
        cout << i << endl;
    }
*/	
}

void  camL()
{
    //VideoCapture cap;
    //cap.open(camURL);
	//string camURL;
	//camURL = "rtsp://admin:pP@697469@192.168.1.102:554/Stream/Channel/101";
	//"rtsp://admin:pP@697469@192"
    namedWindow("Camera2", WINDOW_AUTOSIZE );
    resizeWindow("Camera2", 640, 480);
    moveWindow("Camera2", 700, 0);
/*	
    for (;;) {
       Mat image;
       cap >> image;
       Mat gray_image;
       cvtColor( image, gray_image, COLOR_BGR2GRAY );
	}	
*/
    for( char i = 0; i < 100000; i++){
        lock_guard<mutex> lock(m_console);
        cout << "th::" << i << endl;
    }
}

int main()
{
    // don't call join
	//string cam1_url = "rtsp://root:root@192.168.1.99:554/av0_1";
    //string cam1_url = "rtsp://admin:pP@697469@192.168.1.102:554/Stream/Channel/101" ;
    //string cam2_url = "rtsp://user:9999@192.168.1.167:8557/PSIA/Streaming/channels/2?videoCodecType=H.264";
    //string cam2_url = "rtsp://192.168.1.167:8555/PSIA/Streaming/channels/0?videoCodecType=MJPEG";
    //string filename="/home/yuriy/Videos/ul1.rec";

    thread cam(camExecute,"rtsp://admin:pP@697469@192.168.1.102:554/Stream/Channel/101");
    thread camLL(camL);
    for( int i = 0; i < 100000; i++){
        lock_guard<mutex> lock(m_console);
        cout << "main::" << i << endl;
    }
    // but call there
    camLL.join();
    cam.join();
    system("pause");
    return 0;
}