load("@rules_cc//cc:defs.bzl", "cc_library")

config_setting(
  name = "aarch64-linux-gnu",
  define_values = {"multiarch": "aarch64-linux-gnu"},
)

config_setting(
  name = "x86_64-linux-gnu",
  define_values = {"multiarch": "x86_64-linux-gnu"},
)

cc_library(
  name = "opencv",
  hdrs = glob([
      "opencv4/opencv2/**/*.h*",
  ]) + select({
    ":aarch64-linux-gnu":   ["aarch64-linux-gnu/opencv4/opencv2/cvconfig.h"],
    ":x86_64-linux-gnu":    ["x86_64-linux-gnu/opencv4/opencv2/cvconfig.h"],
    "//conditions:default": [],
  }),
  includes = [
      "opencv4",
  ] + select({
    ":aarch64-linux-gnu":   ["aarch64-linux-gnu/opencv4"],
    ":x86_64-linux-gnu":    ["x86_64-linux-gnu/opencv4"],
    "//conditions:default": [],
  }),
  linkopts = [
    "-l:libopencv_core.so",
    "-l:libopencv_calib3d.so",
    "-l:libopencv_features2d.so",
    "-l:libopencv_highgui.so",
    "-l:libopencv_imgcodecs.so",
    "-l:libopencv_imgproc.so",
    "-l:libopencv_video.so",
    "-l:libopencv_videoio.so",
    "-l:libopencv_dnn.so"
  ],
  visibility = ["//visibility:public"],
)
