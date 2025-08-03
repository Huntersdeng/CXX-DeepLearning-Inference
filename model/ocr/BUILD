load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name="ctc",
    hdrs=["ctc.h", "ocr_model.h"],
    srcs=["ctc.cpp", "ocr_model.cpp"],
    deps=["//model/base:base_model", "@yaml-cpp"],
    visibility = ["//visibility:public"],
)

cc_library(
    name="dbnet",
    hdrs=["dbnet.h", "ocr_model.h"],
    srcs=["dbnet.cpp", "ocr_model.cpp"],
    deps=["//model/base:base_model", "@yaml-cpp", "@polyclipping//:polyclipping"],
    visibility = ["//visibility:public"],
)

cc_library(
    name="attention",
    hdrs=["attention.h", "ocr_model.h"],
    srcs=["attention.cpp", "ocr_model.cpp"],
    deps=["//model/base:base_model", "@yaml-cpp"],
    visibility = ["//visibility:public"],
)