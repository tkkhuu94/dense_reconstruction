#ifndef DENSE_RECONSTRUCTION_RGBD_CAMERA_H_
#define DENSE_RECONSTRUCTION_RGBD_CAMERA_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <string>

class RGBDCamera {
 public:
  inline void SetCameraIntrinsics(double cx, double cy, double fx, double fy) {
    camera_params_.cx = cx;
    camera_params_.cy = cy;
    camera_params_.fx = fx;
    camera_params_.fy = fy;
  }
  inline void SetDepthScale(double depth_scale) {
    camera_params_.depth_scale = depth_scale;
  }
  void BuildPointCloud(const cv::Mat& rgb_image, const cv::Mat& depth_image);

 private:
  struct CameraParams {
    double cx;
    double cy;
    double fx;
    double fy;
    double depth_scale;
  };

  std::string frame_id_;

  CameraParams camera_params_;
};

#endif