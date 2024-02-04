#pragma once
#include <fstream>

#include "model/base/cv_model.h"

class YOLOv8 : public CvModel {
   public:
    YOLOv8() = delete;
    explicit YOLOv8(const std::string &model_path, const std::string framework_type, cv::Size input_size,
                       float conf_thres, float iou_thres);
    explicit YOLOv8(const std::string &yaml_file);
    ~YOLOv8();

    void detect(const cv::Mat &image, std::vector<Object> &objs) override;

   protected:
    void postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) override;

   private:
    cv::Size m_input_size_ = {640, 640};
    float m_conf_thres_ = 0.25f;
    float m_nms_thres_ = 0.65f;
    int topk = 100;
    int strides[3] = {8, 16, 32};
    int m_grid_num_ = 8400;
    PreParam pparam_;
};