load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "base_model",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cpp",
    ]),
    visibility = ["//visibility:public"],
    deps = ["//common", "//framework"],
)
