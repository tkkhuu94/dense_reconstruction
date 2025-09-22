load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "feature_detector_factory",
    srcs = ["feature_detector_factory.cpp"],
    hdrs = ["feature_detector_factory.h"],
    deps = [
        "@opencv",
    ],
)

cc_library(
    name = "camera_pose_estimator",
    srcs = ["camera_pose_estimator.cpp"],
    hdrs = ["camera_pose_estimator.h"],
    deps = [
        ":feature_detector_factory",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/log",
        "@opencv",
    ],
)

cc_binary(
    name = "camera_pose_estimator_binary",
    srcs = ["main.cpp"],
    cxxopts = [
        "-DABSL_MIN_LOG_LEVEL=0",
    ],
    deps = [
        ":camera_pose_estimator",
        ":feature_detector_factory",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:flags",
        "@abseil-cpp//absl/log:initialize",
        "@opencv",
    ],
)
