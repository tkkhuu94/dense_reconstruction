#ifndef VISUAL_ODOMETRY_CAMERA_POSE_ESTIMATOR_H
#define VISUAL_ODOMETRY_CAMERA_POSE_ESTIMATOR_H

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "feature_detector_factory.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

namespace visual_odometry {

class CameraPoseEstimator {
 public:
  static absl::StatusOr<std::unique_ptr<CameraPoseEstimator>> Create(
      const FeatureDetectorType& detector_type,
      cv::Ptr<cv::DescriptorMatcher> matcher);

  ~CameraPoseEstimator() = default;

  absl::StatusOr<cv::Mat> EstimatePose(const cv::Mat& image);

 private:
  CameraPoseEstimator() = default;

  // Stores the last image, keypoints, and descriptors for pose estimation.
  std::vector<cv::KeyPoint> last_keypoints_;
  cv::Mat last_descriptors_;

  cv::Ptr<cv::Feature2D> detector_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;
};
}  // namespace visual_odometry

#endif  // VISUAL_ODOMETRY_CAMERA_POSE_ESTIMATOR_H
