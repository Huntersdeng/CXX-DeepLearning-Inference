#include "common/common.h"
#include <fstream>
#include <sstream>

bool IsFile(const std::string &path)
{
    if (!IsPathExist(path))
    {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool IsFolder(const std::string &path)
{
    if (!IsPathExist(path))
    {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

PreParam Letterbox(const cv::Mat &image, cv::Mat &out, cv::Size size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh)
    {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else
    {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, out, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    PreParam pparam;
    pparam.ratio = 1 / r;
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.height = height;
    pparam.width = width;
    return pparam;
}

PreParam paddimg(const cv::Mat &image, cv::Mat &out, int shortsize) {
    int w = image.cols;
    int h = image.rows;
    float scale = 1.f;
    if (w < h) {
        scale = (float)shortsize / w;
        h = scale * h;
        w = shortsize;
    }
    else {
        scale = (float)shortsize / h;
        w = scale * w;
        h = shortsize;
    }

    if (h % 32 != 0) {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0) {
        w = (w / 32 + 1) * 32;
    }

    cv::resize(image, out, cv::Size(w, h));
    PreParam pparam;
    pparam.ratio = 1 / scale;
    pparam.dw = 0;
    pparam.dh = 0;
    pparam.height = image.rows;
    pparam.width = image.cols;
    return pparam;
}