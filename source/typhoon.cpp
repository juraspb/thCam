#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
int bw = 0;         // Flag BW
int led_pin = 7;    // Pin Definitions
string settingsWin = "Settings";

// Function called by Interrupt
void signalHandler (int s){
  done = true;
}

int camera(string URL)
{
  namedWindow(URL, WINDOW_NORMAL);
  resizeWindow(URL, 640, 480);
  //moveWindow(URL, 0, 0);

  Mat frame,gray;
  VideoCapture cap;
  cap.open(URL);
  while(!done) {
    if (cap.read(frame)) {
      //cout << frame.size()<<endl;
      if (bw==0) {
        //cout << frame.size()<<endl;
        //cvtColor(frame, gray, COLOR_BGR2GRAY);
        //imshow(URL, gray);
        imshow(URL, frame);
      } 
      else {
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        imshow(URL, gray);
      }
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

void on_btn(int position)
{
 if (position==0) {
  // 
  bw = 0;
 }
 else {
  //
  bw = 1;
 }
}

int main()
{
  // don't call join
  w_img.unlock();
  w_cvt.unlock();

  // When CTRL+C pressed, signalHandler will be called to interrupt the programs execution
  signal(SIGINT, signalHandler);

  namedWindow(settingsWin, WINDOW_NORMAL);
  resizeWindow(settingsWin, 320, 120);
  moveWindow(settingsWin, 720, 0);
  createTrackbar("BW", settingsWin, &bw, 1);

  //string URL = "/home/yuriy/Videos/ul1.rec";                                              //from files
  //string URL = "rtsp://admin:pP@697469@192.168.1.102:554/Stream/Channel/101";             //hikvision
	//string URL = "rtsp://root:root@192.168.1.99:554/av0_1";                                 //beward
  //string URL = "rtsp://root:root@192.168.1.12/video_1";                                   //ves-257
  string URL = "rtsp://192.168.11.167:8553/PSIA/Streaming/channels/1?videoCodecType=MPEG4"; // ves-556

  thread cam2(camera,URL);

  int i = 0;
  while(!done)
  {
    this_thread::sleep_for(milliseconds(500));
    cout << i << endl;
    i++;
    if (key=='q') break;
  }  
  // but call there
  cam2.join();
  system("pause");
  destroyAllWindows();
  return 0;
}
