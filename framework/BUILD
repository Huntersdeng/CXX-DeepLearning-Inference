load("@rules_cc//cc:defs.bzl", "cc_library")

config_setting(
    name = "enable_tensorrt",
    values = {"define": "enable_tensorrt=true"},
)

config_setting(
    name = "enable_rknn",
    values = {"define": "enable_rknn=true"},
)

cc_library(
    name = "framework_base",
    hdrs = [
        "config.h",
        "framework.h",
    ],
    defines = select({
                  ":enable_tensorrt": ["USE_TENSORRT"],
                  "//conditions:default": [],
              }) +
              select({
                  ":enable_rknn": ["USE_RKNN"],
                  "//conditions:default": [],
              }),
    visibility = ["//visibility:public"],
    deps = ["//common"],
)

cc_library(
    name = "framework",
    visibility = ["//visibility:public"],
    deps = ["//framework/onnx"] +
           select({
               ":enable_tensorrt": ["//framework/tensorrt"],
               "//conditions:default": [],
           }) +
           select({
               ":enable_rknn": ["//framework/rknn"],
               "//conditions:default": [],
           }),
)
