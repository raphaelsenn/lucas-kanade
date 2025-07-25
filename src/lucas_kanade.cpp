#include <opencv2/opencv.hpp>
#include <vector>
#include "./lucas_kanade.hpp"

// Corner tracking settings
int MAX_TRACKING_CORNERS = 10000;
float QUALITY_TRACKING_CORNERS = 0.001f;
int MIN_DIST_TRACKING_CORNERS = 10;

// Gaussian blur settings for derivative calculation
int GAUSSIAN_BLUR_SIZE = 3;
float GAUSSIAN_BLUR_STD = 0.0f;

// Settings to draw optical flow
float FLOW_ARROW_SCALE = 5.0f;
cv::Scalar ARROW_COLOR(0, 255, 0);
int LINE_THICKNESS = 1;
int LINE_SHIFT = 0;
int LINE_TYPE = cv::LINE_AA;
double TIP_LENGTH = 0.2;

Gradients calculateImageGradients(const cv::Mat& I1, const cv::Mat& I2)
{
  Gradients g;
  cv::Mat I1_gray, I2_gray;
  
  // Convert I1 and I2 to 32-bit grayscale images
  cv::cvtColor(I1, I1_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(I2, I2_gray, cv::COLOR_BGR2GRAY);
  I1_gray.convertTo(I1_gray, CV_32F);
  I2_gray.convertTo(I2_gray, CV_32F);
  
  // Smooth I1_gray and I2_gray for gradient computation 
  cv::GaussianBlur(I1_gray, g.I1_smooth, cv::Size(GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), GAUSSIAN_BLUR_STD);
  cv::GaussianBlur(I2_gray, g.I2_smooth, cv::Size(GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), GAUSSIAN_BLUR_STD);
  
  // Calculate final gradients
  cv::Sobel(g.I1_smooth, g.Ix, CV_32F, 1, 0);
  cv::Sobel(g.I1_smooth, g.Iy, CV_32F, 0, 1);
  g.It = I2_gray - I1_gray;
  return g;
}

cv::Mat lucasKanadeOpticalFlow(const cv::Mat& I1, 
                               const cv::Mat& I2, 
                               const int& kernelSizeLK, 
                               const int& windowWidth, 
                               const int& windowHeight)
{
  // Kernel 'radius'
  int K = kernelSizeLK / 2;
  
  // Calculate image gradients
  Gradients g  = calculateImageGradients(I1, I2);

  // Calculate corners/interesting points 
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(g.I1_smooth, corners, MAX_TRACKING_CORNERS, QUALITY_TRACKING_CORNERS, MIN_DIST_TRACKING_CORNERS);
  
  // Calculating optical flow
  cv::Mat u = cv::Mat::zeros(windowHeight, windowWidth, CV_32F);
  cv::Mat v = cv::Mat::zeros(windowHeight, windowWidth, CV_32F);
  std::vector<float> IX, IY, IT;

  #pragma omp parallel for
  for (std::size_t i = 0; i < corners.size(); i++)
  {
    int x = static_cast<int>(corners[i].x);
    int y = static_cast<int>(corners[i].y);
    IX.clear(); IY.clear(); IT.clear();
    if ((x - K >= 0) && (x + K < windowWidth) && (y - K >= 0) && (y + K < windowHeight))
    {
      for (int w1 = -K; w1 < K+1; w1++)
      {
        for (int w2 = -K; w2 < K+1; w2++)
        {
          // NOTE: not efficient, but works for educational purpose
          IX.push_back(g.Ix.at<float>(y + w2, x + w1));
          IY.push_back(g.Iy.at<float>(y + w2, x + w1));
          IT.push_back(g.It.at<float>(y + w2, x + w1));
        }
      }
      cv::Mat IXMat(IX, true); cv::Mat IYMat(IY, true); cv::Mat ITMat(IT, true); 
      
      // Construct matrix A to solve: Ax = b
      cv::Mat A;
      cv::hconcat(IXMat, IYMat, A);
      
      // <=> (A^T)Ax = (A^T)b
      cv::Mat A_T;
      cv::transpose(A, A_T);
      
      cv::Mat A_T_matmul_A, A_T_matmul_b;
      A_T_matmul_A = A_T * A;
      A_T_matmul_b = A_T * ITMat; // Note: ITMat = b
      
      // <=> x = ((A^TA)^(-1))(A_T)b
      cv::Mat x_sol, A_T_matmul_A_inv;
      cv::invert(A_T_matmul_A, A_T_matmul_A_inv, cv::DECOMP_SVD);
      x_sol = A_T_matmul_A_inv * A_T_matmul_b;
      u.at<float>(y, x) = x_sol.at<float>(0);
      v.at<float>(y, x) = x_sol.at<float>(1);
    }
  }
  
  // Show optical flow
  cv::Mat flow; 
  I1.copyTo(flow);

  #pragma omp parallel for
  for (std::size_t i = 0; i < corners.size(); i++)
  {
    int x = static_cast<float>(corners[i].x);
    int y = static_cast<float>(corners[i].y);
    
    if ((x - K >= 0) && (x + K < windowWidth) && (y - K >= 0) && (y + K < windowHeight))
    {
      float dx = u.at<float>(y, x);
      float dy = v.at<float>(y, x);

      if (std::hypot(dx, dy) < 1.0 || std::hypot(dx, dy) > 10.0) continue;

      cv::Point2f start(x, y);
      cv::Point2f end(x + FLOW_ARROW_SCALE * dx, y + FLOW_ARROW_SCALE * dy);
      cv::arrowedLine(flow, start, end, ARROW_COLOR, LINE_THICKNESS, LINE_TYPE, LINE_SHIFT, TIP_LENGTH);
    }
  }
  return flow;
}