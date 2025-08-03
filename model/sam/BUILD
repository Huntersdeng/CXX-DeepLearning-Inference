load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "sam",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cpp",
    ]),
    visibility = ["//visibility:public"],
    deps = ["//model/base:base_model", "@yaml-cpp"],
)