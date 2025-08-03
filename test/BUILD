load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name="yolo_test",
    srcs=["yolo_test.cpp"],
    deps=["//model/yolo:yolo", "//model/yolo:yolo_cutoff", "//model/yolo:yolo_seg", "//model/yolo:yolo_seg_cutoff", "//model/yolo:yolo_pose"]
)

cc_binary(
    name="ocr_test",
    srcs=["ocr_test.cpp"],
    deps=["//model/ocr:ctc", "//model/ocr:dbnet", "//model/ocr:attention"]
)

cc_binary(
    name="clip_test",
    srcs=["clip_test.cpp"],
    deps=["//model/clip:clip"]
)

cc_binary(
    name="sam_test",
    srcs=["sam_test.cpp"],
    deps=["//model/sam:sam"]
)