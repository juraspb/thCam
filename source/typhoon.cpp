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
#include <sys/socket.h>
#include <netinet/in.h>
#include <udpsocket.hpp>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <netdb.h>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace chrono;

mutex w_out;
mutex w_img;

volatile char key;
bool done = false; // Flag to determine when user wants to end program
int bw = 0;        // Flag BW
int videoON = 0;   // Flag videoON
int fps = 0;
int fps_count = 0;
int pcktCount = 0;
int framesCount = 0;
char framesMemo[1440 * 320 * 3 * 8];
char *framesPtr = framesMemo;
char *frames[8];


int led_pin = 7; // Pin Definitions
string settingsWin = "Settings";

// Function called by Interrupt
void signalHandler(int s)
{
  done = true;
}

int camera(string URL)
{
  namedWindow(URL, WINDOW_NORMAL);
  resizeWindow(URL, 640, 480);
  moveWindow(URL, 340, 0);

  Mat frame, gray;
  VideoCapture cap;
  cap.open(URL);
  while (!done)
  {
    if (cap.read(frame))
    {
      // cout << frame.size()<<endl;
      if (bw == 0)
      {
        // cout << frame.size()<<endl;
        // cvtColor(frame, gray, COLOR_BGR2GRAY);
        // imshow(URL, gray);
        imshow(URL, frame);
      }
      else
      {
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        imshow(URL, gray);
      }
    }
    else
    {
      return -1;
    }
    key = waitKey(1);
    if (key == 'q')
      break;
  }
  destroyAllWindows();
  return 0;
}

int ves250(string URL)
{
  // namedWindow(URL, WINDOW_NORMAL);
  // resizeWindow(URL, 320, 1440);
  // moveWindow(URL, 0, 0);

  // Our constants:
  // const string IP = "192.168.1.46";
  const string IP = "192.168.1.47";
  const uint16_t PORT = 1047;
  char buff[32];

  int sock, length, n;
  socklen_t fromlen;
  struct sockaddr_in server;
  struct sockaddr_in from;

  // Initialize socket.
  sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0)
    cout << PORT << "Error opening socket" << endl; // error("Opening socket");

  length = sizeof(server);
  bzero(&server, length);
  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = htons(PORT);
  if (bind(sock, (struct sockaddr *)&server, length) < 0)
    cout << PORT << "Error binding" << endl;

  fromlen = sizeof(struct sockaddr_in);
  bzero(&from, fromlen);
  from.sin_family = AF_INET;
  from.sin_addr.s_addr = inet_addr("192.168.1.47");
  from.sin_port = htons(PORT);

  // Send String:
  buff[0] = 0x5A;
  buff[1] = 0x86;
  buff[2] = 0x00;
  buff[3] = 0x00;
  buff[4] = 0x00;
  buff[5] = 0x00;
  buff[6] = 0x00;
  buff[7] = buff[0] + buff[1];

  // write(sock, buff, 8);
  int flag = 0;
  sendto(sock, buff, 8, flag, (struct sockaddr *)&from, fromlen);

  while (!done)
  {

    if (frames[framesCount] < 0)
    {
        cout << "Buffer FULL" << endl;
    }
    else
    {

      int n = read(sock, framesPtr, 1500);

      if (n == 48)
      {
        framesPtr += n;
        frames[framesCount]=framesPtr;

        framesCount++;
        framesCount &= 0x07;
        if (framesCount==0) framesPtr=framesMemo;

        w_out.lock();
        cout << "fps::" << fps << "к/c Пакетов:" << pcktCount << endl;
        fps_count++;
        pcktCount = 0;
        w_out.unlock();
      }
      else
      {
        if (n == 1472)
        {
          framesPtr += n;
          pcktCount++;
        }
        else
        {
          cout << "error length::" << n << endl;
        }
      }
    }
    if (key == 'q')
      break;
  }

  // If you want to use std::string:
  /*
  udpSocket.onMessageReceived = [&](string message, string ipv4, uint16_t port) {
      cout << ipv4 << ":" << port << " => " << message << endl;
  };
  */

  return 0;
}

void on_btn(int position)
{
  if (position == 0)
  {
    //
    bw = 0;
  }
  else
  {
    //
    bw = 1;
  }
}

int main()
{
  // don't call join
  w_img.unlock();
  w_out.unlock();

  for (int i = 0; i < 8; i++)
    framesPtr[i] = -1;
  // When CTRL+C pressed, signalHandler will be called to interrupt the programs execution
  signal(SIGINT, signalHandler);

  namedWindow(settingsWin, WINDOW_NORMAL);
  resizeWindow(settingsWin, 320, 120);
  moveWindow(settingsWin, 340, 520);
  createTrackbar("BW", settingsWin, &bw, 1);

  // string URL = "/home/yuriy/Videos/ul1.rec";                                              //from files
  // string URL = "rtsp://admin:pP@697469@192.168.1.102:554/Stream/Channel/101";             //hikvision
  // string URL = "rtsp://root:root@192.168.1.99:554/av0_1";                                 //beward
  // string URL = "rtsp://root:root@192.168.1.12/video_1";                                   //ves-257
  string URL = "rtsp://192.168.1.167:8553/PSIA/Streaming/channels/1?videoCodecType=MPEG4"; // ves-556

  thread cam2(camera, URL);
  thread ves(ves250, "VES250");

  int i = 0;
  while (!done)
  {
    this_thread::sleep_for(milliseconds(1000));
    w_out.lock();
    fps = fps_count;
    fps_count = 0;
    cout << "main:" << i << endl;
    w_out.unlock();
    i++;
    if (key == 'q')
      break;
  }
  // but call there
  cam2.join();
  ves.join();
  this_thread::sleep_for(milliseconds(1000));
  destroyAllWindows();
  return 0;
}

/*

    // You should do an input loop so the program will not terminated immediately:
    string input;
    getline(cin, input);
    while (input != "exit")
    {
        udpSocket.Send(input);
        getline(cin, input);
    }

*/
