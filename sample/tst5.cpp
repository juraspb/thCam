#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
 
cv::Mat img = cv::imread("image.png", IMREAD_GRAYSCALE);
cv::cuda::GpuMat dst, src;
src.upload(img);
 
cv::Ptr<cv::cuda::CLAHE> ptr_clahe = cv::cuda::createCLAHE(5.0, cv::Size(8, 8));
ptr_clahe->apply(src, dst);
 
cv::Mat result;
dst.download(result);
 
cv::imshow("result", result);
cv::waitKey();

