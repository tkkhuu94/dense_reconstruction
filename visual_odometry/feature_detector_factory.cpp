#include "feature_detector_factory.h"

namespace visual_odometry {

cv::Ptr<cv::Feature2D> FeatureDetectorFactory::Create(
    const FeatureDetectorType& detector_type) {
  switch (detector_type) {
    case FeatureDetectorType::kAgast:
      return cv::AgastFeatureDetector::create();
    case FeatureDetectorType::kFast:
      return cv::FastFeatureDetector::create();
    case FeatureDetectorType::kOrb:
      return cv::ORB::create();
    case FeatureDetectorType::kSift:
      return cv::SIFT::create();
    default:
      return nullptr;
  }
}

}  // namespace visual_odometry
