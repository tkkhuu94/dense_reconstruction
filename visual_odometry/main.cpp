#include <algorithm>
#include <cstddef>
#include <filesystem>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "visual_odometry/camera_pose_estimator.h"

void findPathBounds(const std::vector<cv::Mat>& poses, float& min_x,
                    float& max_x, float& min_z, float& max_z) {
  min_x = min_z = std::numeric_limits<float>::max();
  max_x = max_z = std::numeric_limits<float>::lowest();
  for (const auto& pose : poses) {
    // Translation is in the last column of the 4x4 pose matrix.
    // Assumes pose matrix is of type CV_64F (double).
    float x = pose.at<double>(0, 3);
    float z = pose.at<double>(2, 3);
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
  }
}

cv::Mat DrawTrajectory(const std::vector<cv::Mat>& poses) {
  if (poses.empty()) {
    LOG(WARNING) << "Camera poses vector is empty. Cannot draw trajectory.";
    return cv::Mat();
  }

  // Define the canvas size and a margin.
  const int canvas_size = 800;
  const int margin = 50;
  const double axis_scale = 0.05;  // Scale of the drawn coordinate axes.

  // Find the spatial bounds of the path to scale it correctly.
  // We are creating a top-down view, so we use the X and Z coordinates.
  float min_x, max_x, min_z, max_z;
  findPathBounds(poses, min_x, max_x, min_z, max_z);

  float path_width = max_x - min_x;
  float path_height = max_z - min_z;

  // Determine the scaling factor to fit the path onto the canvas.
  float scale = 1.0f;
  if (path_width > 1e-6 && path_height > 1e-6) {  // Avoid division by zero
    scale = std::min((float)(canvas_size - 2 * margin) / path_width,
                     (float)(canvas_size - 2 * margin) / path_height);
  }

  // Create a blank white canvas.
  cv::Mat canvas = cv::Mat::ones(canvas_size, canvas_size, CV_8UC3) * 255;

  // Draw the path by connecting consecutive positions.
  for (size_t i = 0; i < poses.size() - 1; ++i) {
    cv::Point3f pos1(poses[i].at<double>(0, 3), poses[i].at<double>(1, 3),
                     poses[i].at<double>(2, 3));
    cv::Point3f pos2(poses[i + 1].at<double>(0, 3),
                     poses[i + 1].at<double>(1, 3),
                     poses[i + 1].at<double>(2, 3));

    // Transform world coordinates (X, Z) to image coordinates (u, v).
    cv::Point2f p1(margin + (pos1.x - min_x) * scale,
                   margin + (pos1.z - min_z) * scale);
    cv::Point2f p2(margin + (pos2.x - min_x) * scale,
                   margin + (pos2.z - min_z) * scale);

    // Draw a light grey line for the trajectory path.
    cv::line(canvas, p1, p2, cv::Scalar(180, 180, 180), 1);
  }

  // Draw a coordinate frame for each pose.
  for (const auto& pose : poses) {
    // Extract rotation and translation from the 4x4 pose matrix.
    cv::Mat R = pose(cv::Rect(0, 0, 3, 3));
    cv::Mat t = pose(cv::Rect(3, 0, 1, 3));

    // Define local coordinate axes.
    cv::Mat x_axis = (cv::Mat_<double>(3, 1) << axis_scale, 0, 0);
    cv::Mat z_axis = (cv::Mat_<double>(3, 1) << 0, 0, axis_scale);

    // Rotate axes into world coordinates.
    cv::Mat x_world = R * x_axis;
    cv::Mat z_world = R * z_axis;

    // Project origin and axis endpoints to the 2D canvas (top-down view).
    cv::Point2f origin_2d(margin + (t.at<double>(0) - min_x) * scale,
                          margin + (t.at<double>(2) - min_z) * scale);

    cv::Point2f x_end_2d(
        margin + (t.at<double>(0) + x_world.at<double>(0) - min_x) * scale,
        margin + (t.at<double>(2) + x_world.at<double>(2) - min_z) * scale);

    cv::Point2f z_end_2d(
        margin + (t.at<double>(0) + z_world.at<double>(0) - min_x) * scale,
        margin + (t.at<double>(2) + z_world.at<double>(2) - min_z) * scale);

    // Draw the Z-axis (camera's forward direction) in blue.
    cv::line(canvas, origin_2d, z_end_2d, cv::Scalar(255, 0, 0), 2);
    // Draw the X-axis (camera's right direction) in red.
    cv::line(canvas, origin_2d, x_end_2d, cv::Scalar(0, 0, 255), 2);
  }

  return canvas;
}

ABSL_FLAG(std::string, image_directory, "", "Path to the image directory");
ABSL_FLAG(std::string, visualized_image, "", "Path to save the visualized image");

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);

  LOG(INFO) << "Starting Visual Odometry...";

  if (absl::GetFlag(FLAGS_image_directory).empty()) {
    LOG(ERROR) << "Image directory path is required.";
    return 1;
  }

  std::filesystem::path image_dir = absl::GetFlag(FLAGS_image_directory);

  if (!std::filesystem::exists(image_dir) ||
      !std::filesystem::is_directory(image_dir)) {
    LOG(ERROR) << "Invalid image directory: " << image_dir;
    return 1;
  }

  std::vector<std::filesystem::path> image_paths;
  for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
    if (entry.is_regular_file() && (entry.path().extension() == ".jpg" ||
                                    entry.path().extension() == ".png" ||
                                    entry.path().extension() == ".jpeg")) {
      image_paths.emplace_back(entry.path());
    }
  }

  std::sort(image_paths.begin(), image_paths.end());

  if (image_paths.empty()) {
    LOG(ERROR) << "No images found in directory: " << image_dir;
    return 1;
  }

  LOG(INFO) << "Found " << image_paths.size() << " images in directory.";

  auto estimator = visual_odometry::CameraPoseEstimator::Create(
      visual_odometry::FeatureDetectorType::kOrb,
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE));
  if (!estimator.ok()) {
    LOG(ERROR) << "Error creating CameraPoseEstimator: "
               << estimator.status().message();
    return 1;
  }

  LOG(INFO) << "CameraPoseEstimator created successfully.";

  std::vector<cv::Mat> camera_poses;

  size_t image_count = 0;
  for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
      LOG(WARNING) << "Failed to read image: " << image_path;
      continue;
    }

    auto pose = (*estimator)->EstimatePose(image);
    if (!pose.ok()) {
      LOG(WARNING) << "Pose estimation failed for image " << image_path << ": "
                   << pose.status().message();
      continue;
    }

    camera_poses.emplace_back(pose.value());
    image_count++;
    if (image_count % 10 == 0) {
      LOG(INFO) << "Processed " << image_count << " images.";
    }

    if (image_count >= 50) {
      // Limit to first 100 images for efficiency.
      break;
    }
  }

  LOG(INFO) << "Processed " << camera_poses.size() << " images.";

  cv::Mat trajectory_image = DrawTrajectory(camera_poses);
  if (!trajectory_image.empty()) {
    cv::imwrite(absl::GetFlag(FLAGS_visualized_image).c_str(), trajectory_image);
    LOG(INFO)
        << "Saved camera trajectory visualization to camera_trajectory.png";
  } else {
    LOG(WARNING) << "Failed to create trajectory visualization.";
  }

  return 0;
}
