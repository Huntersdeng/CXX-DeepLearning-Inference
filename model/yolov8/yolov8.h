#pragma once
#include "model/base/cv_model.h"
#include <fstream>

class YOLOv8Seg : public CvModel
{
public:
    explicit YOLOv8Seg(const std::string &engine_file_path);
    explicit YOLOv8Seg(const std::string &engine_file_path, cv::Size input_size, float conf_thres, float iou_thres, cv::Size seg_size, int seg_channels);
    ~YOLOv8Seg();

protected:
    void postprocess(const std::vector<void *> output, std::vector<det::Object> &objs);

    cv::Size m_seg_size_ = {160, 160};
    int m_seg_channels_ = 32;
    float m_score_thres_ = 0.25f;
    float m_iou_thres_ = 0.65f;
    int topk = 100;
};  