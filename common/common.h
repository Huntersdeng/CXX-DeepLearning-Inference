#pragma once
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <unistd.h>

struct Binding
{
    size_t size = 1;
    size_t dsize = 1;
    std::vector<int64_t> dims;
    std::string name;
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

PreParam Letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);

PreParam paddimg(const cv::Mat &image, cv::Mat &out, int shortsize = 960);