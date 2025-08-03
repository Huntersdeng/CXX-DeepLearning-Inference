load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
  name = "polyclipping",
  hdrs = ["polyclipping/clipper.hpp"],
  linkopts = [
    "-l:libpolyclipping.so"
  ],
  visibility = ["//visibility:public"],
)