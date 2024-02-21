#pragma once
#include "model/base/model.h"
#include "common/common.h"
#include "framework/config.h"

#ifdef USE_TENSORRT
    #include "framework/tensorrt/tensorrt.h"
#endif

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189}, {217, 83, 25}, {237, 177, 32}, {126, 47, 142}, {119, 172, 48}, {77, 190, 238}, {162, 20, 47}, {76, 76, 76}, {153, 153, 153}, {255, 0, 0}, {255, 128, 0}, {191, 191, 0}, {0, 255, 0}, {0, 0, 255}, {170, 0, 255}, {85, 85, 0}, {85, 170, 0}, {85, 255, 0}, {170, 85, 0}, {170, 170, 0}, {170, 255, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {0, 85, 128}, {0, 170, 128}, {0, 255, 128}, {85, 0, 128}, {85, 85, 128}, {85, 170, 128}, {85, 255, 128}, {170, 0, 128}, {170, 85, 128}, {170, 170, 128}, {170, 255, 128}, {255, 0, 128}, {255, 85, 128}, {255, 170, 128}, {255, 255, 128}, {0, 85, 255}, {0, 170, 255}, {0, 255, 255}, {85, 0, 255}, {85, 85, 255}, {85, 170, 255}, {85, 255, 255}, {170, 0, 255}, {170, 85, 255}, {170, 170, 255}, {170, 255, 255}, {255, 0, 255}, {255, 85, 255}, {255, 170, 255}, {85, 0, 0}, {128, 0, 0}, {170, 0, 0}, {212, 0, 0}, {255, 0, 0}, {0, 43, 0}, {0, 85, 0}, {0, 128, 0}, {0, 170, 0}, {0, 212, 0}, {0, 255, 0}, {0, 0, 43}, {0, 0, 85}, {0, 0, 128}, {0, 0, 170}, {0, 0, 212}, {0, 0, 255}, {0, 0, 0}, {36, 36, 36}, {73, 73, 73}, {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189}, {80, 183, 189}, {128, 128, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56}, {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49}, {72, 249, 10}, {146, 204, 23}, {61, 219, 134}, {26, 147, 52}, {0, 212, 187}, {44, 153, 168}, {0, 194, 255}, {52, 69, 147}, {100, 115, 255}, {0, 24, 236}, {132, 56, 255}, {82, 0, 133}, {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};

void ReadClassNames(std::string file_name, std::vector<std::string> &class_names);

struct Object
{
    cv::Rect_<float> rect;
    int label = 0;
    float prob = 0.0;
    cv::Mat boxMask;
};

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

void DrawBoxes(const cv::Mat &image,
                  cv::Mat &res,
                  const std::vector<Object> &objs);

float Iou(cv::Rect bb_test, cv::Rect bb_gt);

void Nms(std::vector<Object> &res, float nms_thresh);

class DetectionModel : public Model
{
public:
    explicit DetectionModel() {}; 
    virtual ~DetectionModel() {};
    virtual void detect(const cv::Mat &image, std::vector<Object> &objs) = 0;
protected:
    virtual void preprocess(const cv::Mat &input_image, cv::Mat &output_image) = 0;
    virtual void postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) = 0;
};