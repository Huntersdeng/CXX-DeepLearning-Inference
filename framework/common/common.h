#pragma once
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(call)                                                         \
    do                                                                      \
    {                                                                       \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// const std::vector<std::string> CLASS_NAMES = {
//     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
//     "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
//     "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
//     "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
//     "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
//     "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
//     "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
//     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
//     "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
//     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
//     "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
//     "teddy bear", "hair drier", "toothbrush"};

// const std::vector<std::string> CLASS_NAMES = {"paper", "mask", "thread", "obstacle", "bottle", "blood", "person"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189}, {217, 83, 25}, {237, 177, 32}, {126, 47, 142}, {119, 172, 48}, {77, 190, 238}, {162, 20, 47}, {76, 76, 76}, {153, 153, 153}, {255, 0, 0}, {255, 128, 0}, {191, 191, 0}, {0, 255, 0}, {0, 0, 255}, {170, 0, 255}, {85, 85, 0}, {85, 170, 0}, {85, 255, 0}, {170, 85, 0}, {170, 170, 0}, {170, 255, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {0, 85, 128}, {0, 170, 128}, {0, 255, 128}, {85, 0, 128}, {85, 85, 128}, {85, 170, 128}, {85, 255, 128}, {170, 0, 128}, {170, 85, 128}, {170, 170, 128}, {170, 255, 128}, {255, 0, 128}, {255, 85, 128}, {255, 170, 128}, {255, 255, 128}, {0, 85, 255}, {0, 170, 255}, {0, 255, 255}, {85, 0, 255}, {85, 85, 255}, {85, 170, 255}, {85, 255, 255}, {170, 0, 255}, {170, 85, 255}, {170, 170, 255}, {170, 255, 255}, {255, 0, 255}, {255, 85, 255}, {255, 170, 255}, {85, 0, 0}, {128, 0, 0}, {170, 0, 0}, {212, 0, 0}, {255, 0, 0}, {0, 43, 0}, {0, 85, 0}, {0, 128, 0}, {0, 170, 0}, {0, 212, 0}, {0, 255, 0}, {0, 0, 43}, {0, 0, 85}, {0, 0, 128}, {0, 0, 170}, {0, 0, 212}, {0, 0, 255}, {0, 0, 0}, {36, 36, 36}, {73, 73, 73}, {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189}, {80, 183, 189}, {128, 128, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56}, {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49}, {72, 249, 10}, {146, 204, 23}, {61, 219, 134}, {26, 147, 52}, {0, 212, 187}, {44, 153, 168}, {0, 194, 255}, {52, 69, 147}, {100, 115, 255}, {0, 24, 236}, {132, 56, 255}, {82, 0, 133}, {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};

void ReadClassNames(std::string file_name, std::vector<std::string> &class_names);

struct Binding
{
    size_t size = 1;
    size_t dsize = 1;
    std::vector<int64_t> dims;
    std::string name;
};

struct Object
{
    cv::Rect_<float> rect;
    int label = 0;
    float prob = 0.0;
    cv::Mat boxMask;
};

struct PreParam
{
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
    float height = 0;
    float width = 0;
};

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string &path)
{
    return (access(path.c_str(), 0) == F_OK);
}

bool IsFile(const std::string &path);

bool IsFolder(const std::string &path);

void DrawObjects(const cv::Mat &image,
                  cv::Mat &res,
                  const std::vector<Object> &objs,
                  const std::vector<std::string> &CLASS_NAMES,
                  const std::vector<std::vector<unsigned int>> &COLORS);

void DrawObjectsMasks(const cv::Mat &image,
                        cv::Mat &res,
                        const std::vector<Object> &objs,
                        const std::vector<std::string> &CLASS_NAMES,
                        const std::vector<std::vector<unsigned int>> &COLORS,
                        const std::vector<std::vector<unsigned int>> &MASK_COLORS);

float Iou(cv::Rect bb_test, cv::Rect bb_gt);

void Nms(std::vector<Object> &res, float nms_thresh);

PreParam Letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);