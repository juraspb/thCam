#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <fstream>
#include <unistd.h>
#include <signal.h>
#include <chrono>
//#include <JetsonGPIO.h>

using namespace std;
using namespace cv;
using namespace chrono;

mutex w_cvt;
mutex w_img;

volatile char key;
bool done = false;  // Flag to determine when user wants to end program
int led_pin = 7;    // Pin Definitions

// Function called by Interrupt
void signalHandler (int s){
  done = true;
}

int camera(string URL)
{
  namedWindow(URL, WINDOW_AUTOSIZE);
  //resizeWindow(URL, 640, 480);
  //moveWindow(URL, 0, 0);

  Mat frame,gray;
  VideoCapture cap;
  cap.open(URL);
  while(!done) {
    if (cap.read(frame)) {
      //cout << frame.size()<<endl;
      cvtColor(frame, gray, COLOR_BGR2GRAY);
      imshow(URL, gray);
    }	
    else {
      return -1;
    }
    key = waitKey(1);
 	  if (key=='q') break;
  } 
  destroyAllWindows();
  return 0;
}



int main()
{
  // don't call join
  w_img.unlock();
  w_cvt.unlock();

  // When CTRL+C pressed, signalHandler will be called to interrupt the programs execution
  signal(SIGINT, signalHandler);

	//string cam1_url = "rtsp://root:root@192.168.1.99:554/av0_1";
  //string cam1_url = "rtsp://admin:pP@697469@192.168.1.102:554/Stream/Channel/101" ;
  //string cam2_url = "rtsp://user:9999@192.168.1.167:8557/PSIA/Streaming/channels/2?videoCodecType=H.264";
  //string filename="/home/yuriy/Videos/ul1.rec";
  

  //string URL = "rtsp://root:root@192.168.1.12/video_1";
  //string URL = "rtsp://192.168.11.167:8555/PSIA/Streaming/channels/0?videoCodecType=MJPEG";
  //string URL = "rtsp://192.168.11.167:8553/PSIA/Streaming/channels/1?videoCodecType=MPEG4";
  string URL = "rtsp://admin:pP@697469@192.168.1.102:554/Stream/Channel/101";

  thread cam2(camera,URL);

  //namedWindow(URL, WINDOW_AUTOSIZE);
  //moveWindow(URL, 0, 0);
  //namedWindow(URL2, WINDOW_AUTOSIZE);
  //moveWindow(URL2, 800, 0);
  //Mat frame,gray;
  //VideoCapture cap;
  //cap.open(URL);
  //VideoCapture cap2;
  //cap2.open(URL2);
  int i = 0;
  while(!done)
  {
    this_thread::sleep_for(milliseconds(500));
    cout << i << endl;
    i++;


/*
    if (cap.read(frame)) 
    {
      //cout << frame.size()<<endl;
      //cvtColor(frame, gray, COLOR_BGR2GRAY);
      imshow(URL, frame);
    }	
    else
    {
      return -1;
    }

    if (cap2.read(frame)) 
    {
      //cout << frame.size()<<endl;
      cvtColor(frame, gray, COLOR_BGR2GRAY);
      imshow(URL2, gray);
    }	
    else
    {
      return -1;
    }
*/    
    if (key=='q') break;
  }  
  // but call there
  cam2.join();
  system("pause");
  destroyAllWindows();
  return 0;
}
