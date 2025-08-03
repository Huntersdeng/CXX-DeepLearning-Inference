load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

cc_import(
    name = "onnxruntime_shared",
    shared_library = "lib/libonnxruntime.so.1.16.2",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "onnxruntime_headers",
    hdrs = glob(["include/*.h"]),  # 包含所有头文件
    includes = ["include"],  # 头文件搜索路径设为当前目录,
    strip_include_prefix="include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "onnxruntime",
    visibility = ["//visibility:public"],
    deps = [
        ":onnxruntime_headers",
        ":onnxruntime_shared",
    ],
)
