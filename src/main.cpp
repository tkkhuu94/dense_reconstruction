#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

#include "RGBDCamera.h"

int main(int argc, char** argv) {
  RGBDCamera rgbd_camera;
  rgbd_camera.SetCameraIntrinsics(0, 0, 0, 0);
  return 0;
}