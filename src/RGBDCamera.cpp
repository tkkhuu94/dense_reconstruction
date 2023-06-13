#include "RGBDCamera.h"

#include <vector>

void RGBDCamera::BuildPointCloud(const cv::Mat& rgb_image,
                                 const cv::Mat& depth_image) {

  
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width = static_cast<uint32_t>(depth_image.cols) *
          static_cast<uint32_t>(depth_image.rows);
    cloud.height = 1;
  cloud.is_dense = true;  // We are not storing any points with NaN values


  int n_rows = depth_image.rows;
  int n_cols = depth_image.cols;

  for (int row = 0; row < n_rows; row++) {
    for (int col = 0; col < n_cols; col++) {
        double depth = static_cast<double>(depth_image.at<uint16_t>(row, col));
        if (depth == 0) {
            continue;
        }

        double z = depth / camera_params_.depth_scale;
        double x = (row - camera_params_.cx) * z / camera_params_.fx;
        double y = (col - camera_params_.cy) * z / camera_params_.fy;
    }
  }
  while (rgb_image_it != rgb_image.end() && depth_image_it != rgb_image.end()) {
    
    cloud.push_back(

    );

  }
}