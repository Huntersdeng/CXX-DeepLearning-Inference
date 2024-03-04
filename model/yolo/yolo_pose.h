#pragma once
#include <fstream>

#include "model/base/detection_model.h"

class YOLOPose : public DetectionModel {
   public:
    YOLOPose() = delete;
    explicit YOLOPose(const std::string &yaml_file);
    ~YOLOPose();

    void detect(const cv::Mat &image, std::vector<Object> &objs) override;

   protected:
    void preprocess(const cv::Mat &input_image, cv::Mat &output_image) override;
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