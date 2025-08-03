load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "onnx",
    hdrs=["onnx.h"],
    srcs=["onnx.cpp"],
    deps=[
        "@onnxruntime//:onnxruntime",
        "//framework:framework_base"
    ],
    visibility = ["//visibility:public"],
)