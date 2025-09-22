#include "camera_pose_estimator.h"

#include "opencv2/calib3d.hpp"
#include "absl/log/log.h"

namespace visual_odometry {

absl::StatusOr<std::unique_ptr<CameraPoseEstimator>>
CameraPoseEstimator::Create(const FeatureDetectorType& detector_type,
                            cv::Ptr<cv::DescriptorMatcher> matcher) {
  if (matcher == nullptr) {
    return absl::InvalidArgumentError("Matcher is null");
  }

  auto detector = FeatureDetectorFactory::Create(detector_type);
  if (detector == nullptr) {
    return absl::InvalidArgumentError("Detector is null");
  }

  auto estimator =
      std::unique_ptr<CameraPoseEstimator>(new CameraPoseEstimator());

  if (estimator == nullptr) {
    return absl::InternalError("Failed to create CameraPoseEstimator");
  }

  estimator->detector_ = std::move(detector);
  estimator->matcher_ = std::move(matcher);
  return estimator;
}

absl::StatusOr<cv::Mat> CameraPoseEstimator::EstimatePose(
    const cv::Mat& image) {
  if (image.empty()) {
    return absl::InvalidArgumentError("Input image is empty");
  }

  if (last_keypoints_.empty() || last_descriptors_.empty()) {
    try {
      detector_->detectAndCompute(image, cv::noArray(), last_keypoints_,
                                  last_descriptors_);
    } catch (const cv::Exception& e) {
      return absl::InternalError(
          std::string("OpenCV error during detectAndCompute: ") + e.what());
    }
    return cv::Mat::eye(4, 4, CV_64F);
  }

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  detector_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

  if (keypoints.empty()) {
    return absl::InternalError("Failed to detect keypoints");
  }

  if (descriptors.empty()) {
    return absl::InternalError("Failed to compute descriptors");
  }

  std::vector<cv::DMatch> matches;
  matcher_->match(last_descriptors_, descriptors, matches);

  if (matches.size() < 8) {
    return absl::InternalError("Not enough matches found");
  }

  std::vector<cv::Point2f> points1, points2;
  for (const auto& match : matches) {
    points1.emplace_back(last_keypoints_[match.queryIdx].pt);
    points2.emplace_back(keypoints[match.trainIdx].pt);
  }

  auto essential_matrix = cv::findEssentialMat(points1, points2, cv::RANSAC);

  if (essential_matrix.empty()) {
    return absl::InternalError("Failed to compute essential matrix");
  }

  cv::Mat R, t;
  int inliers = cv::recoverPose(essential_matrix, points1, points2, R, t);

  if (inliers < 8) {
    return absl::InternalError("Not enough inliers after pose recovery");
  }

  cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  t.copyTo(pose(cv::Rect(3, 0, 1, 3)));
  R.copyTo(pose(cv::Rect(0, 0, 3, 3)));

  return pose;
}

}  // namespace visual_odometry
