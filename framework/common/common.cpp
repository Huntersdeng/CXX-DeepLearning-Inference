#include "common/common.h"
#include <fstream>
#include <sstream>

void ReadClassNames(std::string file_name, std::vector<std::string> &class_names)
{
    std::ifstream in_file;
    in_file.open(file_name, std::ios::in);
    assert(in_file.good());

    std::string name;
    while (getline(in_file, name, '\n'))
    {
        class_names.push_back(name);
    }
    in_file.close();
}

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

void DrawObjects(const cv::Mat &image,
                  cv::Mat &res,
                  const std::vector<Object> &objs,
                  const std::vector<std::string> &CLASS_NAMES,
                  const std::vector<std::vector<unsigned int>> &COLORS)
{
    res = image.clone();
    for (auto &obj : objs)
    {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

void DrawObjectsMasks(const cv::Mat &image,
                        cv::Mat &res,
                        const std::vector<Object> &objs,
                        const std::vector<std::string> &CLASS_NAMES,
                        const std::vector<std::vector<unsigned int>> &COLORS,
                        const std::vector<std::vector<unsigned int>> &MASK_COLORS)
{
    res = image.clone();
    cv::Mat mask = image.clone();
    for (auto &obj : objs)
    {
        int idx = obj.label;
        cv::Scalar color = cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
        cv::Scalar mask_color =
            cv::Scalar(MASK_COLORS[idx % 20][0], MASK_COLORS[idx % 20][1], MASK_COLORS[idx % 20][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[idx].c_str(), obj.prob * 100);
        mask(obj.rect).setTo(mask_color, obj.boxMask);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
}

float Iou(cv::Rect bb_test, cv::Rect bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return in / un;
}

void Nms(std::vector<Object> &res, float nms_thresh)
{
    std::map<float, std::vector<Object>> m;
    for (const auto &obj : res)
    {
        if (m.count(obj.label) == 0)
        {
            m.emplace(obj.label, std::vector<Object>());
        }
        m[obj.label].push_back(obj);
    }
    auto cmp = [](const Object &a, const Object &b)
    {
        return a.prob > b.prob;
    };
    res.clear();
    for (auto it = m.begin(); it != m.end(); it++)
    {
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m)
        {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n)
            {
                if (Iou(item.rect, dets[n].rect) > nms_thresh)
                {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

PreParam Letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size)
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

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    PreParam pparam;
    pparam.ratio = 1 / r;
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.height = height;
    pparam.width = width;
    return pparam;
}