#ifndef VISUAL_ODOMETRY_FEATURE_FEATURE_DETECTOR_FACTORY_
#define VISUAL_ODOMETRY_FEATURE_FEATURE_DETECTOR_FACTORY_

#include "opencv2/features2d.hpp"

namespace visual_odometry {

enum class FeatureDetectorType {
  kUnknown = 0,
  kAgast,
  kFast,
  kOrb,
  kSift,
};

class FeatureDetectorFactory {
 public:
  static cv::Ptr<cv::Feature2D> Create(
      const FeatureDetectorType& detector_type);
};

}  // namespace visual_odometry

#endif
