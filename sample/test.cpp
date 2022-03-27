
#include <iostream>
#include <iomanip>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "arcos.h"
#include "cuda.h"
#include "cuda_runtime.h"


using namespace std;
using namespace cv;
using namespace cv::cuda;



void help()
{
    cout << "Usage: ./cascadeclassifier \n\t--cascade <cascade_file>\n\t(<image>|--video <video>|--camera <camera_id>)\n"
        "Using OpenCV version " << CV_VERSION << endl << endl;
}



void convertAndResize(const Mat& src, Mat& gray, Mat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = src;
    }



    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));



    if (scale != 1)
    {
        cv::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}



void convertAndResize(const GpuMat& src, GpuMat& gray, GpuMat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = src;
    }



    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));



    if (scale != 1)
    {
        cv::cuda::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}



void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);



    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, Scalar(0, 0, 0), 5 * fontThickness / 2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}



void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorRed = Scalar(255, 0, 0);
    Scalar fontColorNV = Scalar(118, 185, 0);



    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        (bGpu ? "GPU, " : "CPU, ") <<
        (bLargestFace ? "One, " : "Multi, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");
    matPrint(canvas, 1, fontColorRed, ss.str());



    // by Anatoly. MacOS fix. ostringstream(const string&) is a private
    // matPrint(canvas, 2, fontColorNV, ostringstream("Space - switch GPU / CPU"));
    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "Space - switch GPU / CPU");
        matPrint(canvas, 3, fontColorNV, "M - switch One / Multi");
        matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
        matPrint(canvas, 6, fontColorNV, "1/Q - increase/decrease scale");
        matPrint(canvas, 7, fontColorNV, "R - rotate on/off");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
}



void elforgat(Mat& src, double fok)
{
    Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    Mat rot = cv::getRotationMatrix2D(center, fok, 1.0);
    Mat f;
    cv::warpAffine(src, f, rot, src.size());
    src = f;
}
void elforgat(GpuMat& src, double fok)
{
    Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    Mat rot = cv::getRotationMatrix2D(center, fok, 1.0);
    GpuMat f;
    cuda::warpAffine(src, f, rot, src.size());
    src = f;
}



__global__ void detectLyme(cv::Ptr<cv::cuda::CascadeClassifier> &cascade_gpu, bool findLargestObject, double scaleFactor, bool filterRects, cv::Size &minSize, cv::Size &maxSize, cv::cuda::GpuMat &resized_gpu, cv::cuda::GpuMat &facesBuf_gpu, std::vector<cv::Rect> &faces)
{
    cascade_gpu->setFindLargestObject(findLargestObject);
    cascade_gpu->setScaleFactor(1 / scaleFactor);
    cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
    cascade_gpu->setMinObjectSize(minSize);
    cascade_gpu->setMaxObjectSize(maxSize);
    cascade_gpu->detectMultiScale(resized_gpu, facesBuf_gpu);
    cascade_gpu->convert(facesBuf_gpu, faces);
}



int main(int argc, const char *argv[])
{
    VideoCapture capture;
    Mat image;



    string inputName;
    bool isInputImage = false;
    bool isInputVideo = true;
    bool isInputCamera = false;
    Ptr<cuda::CascadeClassifier> cascade_gpu;
    cv::CascadeClassifier cascade_cpu;
    Mat frame, frame_cpu, gray_cpu, resized_cpu, frameDisp;
    vector<Rect> faces;
    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;
    bool useGPU = true;
    double scaleFactor = 0.9;
    bool findLargestObject = false;
    bool filterRects = true;
    bool forgass = false;
    bool helpScreen = false, run = true;
    double fok = 0;
    TickMeter tm;
    Size minSize = Size(30, 30);
    Size maxSize = Size(34, 34);
    double detectionTime, fps;
    char key;
    string cascadeName = "cascade.xml";



    inputName = "D:\\cicc\\video\\085\\Teszt40\\Teszt_Teszt40_2018-10-09-17-09-25_free_1.avi";



    setlocale(LC_ALL, "hun");



    if (getCudaEnabledDeviceCount() == 0)
        return cerr << "No GPU found or the library is compiled without CUDA support" << endl, -1;



    //cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    cuda::printCudaDeviceInfo(cv::cuda::getDevice());



    cascade_gpu = cuda::CascadeClassifier::create(cascadeName);
    cout << "name: " << cascade_gpu->getDefaultName() << " size " << cascade_gpu->getClassifierSize() << endl;



    if (!cascade_cpu.load(cascadeName))
        return cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"" << endl, help(), -1;



    if (isInputImage)
    {
        image = imread(inputName);
        CV_Assert(!image.empty());
    }
    else if (isInputVideo)
    {
        capture.open(inputName);
        CV_Assert(capture.isOpened());
    }
    else
    {
        capture.open(atoi(inputName.c_str()));
        CV_Assert(capture.isOpened());
    }
    namedWindow("lajm-e", 1);
    while (run)
    {
        if (isInputCamera || isInputVideo)
        {
            capture >> frame;
            if (frame.empty()) run = false;
        }
        if (run) // end ?
        {
            tm.reset();
            if (useGPU)
            {
                tm.start();
                frame_gpu.upload(image.empty() ? frame : image);
                if (forgass) elforgat(frame_gpu, fok);
                convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
                //detectLyme(cascade_gpu, findLargestObject, scaleFactor, filterRects, minSize, maxSize, resized_gpu, facesBuf_gpu, faces);
                detectLyme<<<1, 256>>>(cascade_gpu, findLargestObject, scaleFactor, filterRects, minSize, maxSize, resized_gpu, facesBuf_gpu, faces);
                tm.stop();
                resized_gpu.download(resized_cpu);
            }
            else
            {
                tm.start();
                (image.empty() ? frame : image).copyTo(frame_cpu);
                convertAndResize(frame_cpu, gray_cpu, resized_cpu, scaleFactor);
                if (forgass) elforgat(resized_cpu, fok);
                cascade_cpu.detectMultiScale(resized_cpu, faces, 1.05,
                    (filterRects || findLargestObject) ? 4 : 0,
                    (findLargestObject ? CASCADE_FIND_BIGGEST_OBJECT : 0)
                    | CASCADE_SCALE_IMAGE
                    , minSize, maxSize);
                tm.stop();
            }
            if (forgass) fok++;
            for (size_t i = 0; i < faces.size(); ++i)
                rectangle(resized_cpu, faces[i], Scalar(255));



            detectionTime = tm.getTimeMilli();
            fps = 1000 / detectionTime;



            //print detections to console
            cout << setfill(' ') << setprecision(2);
            cout << setw(6) << fixed << fps << " FPS, " << faces.size() << " det";
            if ((filterRects || findLargestObject) && !faces.empty())
                for (size_t i = 0; i < faces.size(); ++i)
                    cout << ", [" << setw(4) << faces[i].x
                    << ", " << setw(4) << faces[i].y
                    << ", " << setw(4) << faces[i].width
                    << ", " << setw(4) << faces[i].height << "]";
            cout << endl;



            cv::cvtColor(resized_cpu, frameDisp, COLOR_GRAY2BGR);
            displayState(frameDisp, helpScreen, useGPU, findLargestObject, filterRects, fps);
            imshow("lajm-e", frameDisp);



            key = (char)waitKey(1);



            switch (key)
            {
            case 27:
                run = false; break;
            case ' ':
                useGPU = !useGPU; break;
            case 'm':   case 'M':
                findLargestObject = !findLargestObject; break;
            case 'f':   case 'F':
                filterRects = !filterRects; break;
            case '1':
                scaleFactor *= 1.05; break;
            case 'q':   case 'Q':
                scaleFactor /= 1.05; break;
            case 'h':   case 'H':
                helpScreen = !helpScreen; break;
            case 'r':   case 'R':
                forgass = !forgass; break;
            }
        } // if run
    } // while (run)
    cout << "File v�ge" << endl;
    //if (key!=27) std::cin.get();
    return 0;
}