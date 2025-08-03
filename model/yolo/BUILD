load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name="yolo",
    hdrs=["yolo.h"],
    srcs=["yolo.cpp"],
    deps=["//model/base:base_model", "@yaml-cpp"],
    visibility = ["//visibility:public"],
)

cc_library(
    name="yolo_cutoff",
    hdrs=["yolo_cutoff.h"],
    srcs=["yolo_cutoff.cpp"],
    deps=["//model/base:base_model", "@yaml-cpp"],
    visibility = ["//visibility:public"],
)

cc_library(
    name="yolo_seg",
    hdrs=["yolo_seg.h"],
    srcs=["yolo_seg.cpp"],
    deps=["//model/base:base_model", "@yaml-cpp"],
    visibility = ["//visibility:public"],
)

cc_library(
    name="yolo_seg_cutoff",
    hdrs=["yolo_seg_cutoff.h"],
    srcs=["yolo_seg_cutoff.cpp"],
    deps=["//model/base:base_model", "@yaml-cpp"],
    visibility = ["//visibility:public"],
)

cc_library(
    name="yolo_pose",
    hdrs=["yolo_pose.h"],
    srcs=["yolo_pose.cpp"],
    deps=["//model/base:base_model", "@yaml-cpp"],
    visibility = ["//visibility:public"],
)