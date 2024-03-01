#pragma once
#include <fstream>

#include "model/base/detection_model.h"

class YOLO : public DetectionModel {
   public:
    YOLO() = delete;
    explicit YOLO(const std::string &yaml_file);
    ~YOLO();

    void detect(const cv::Mat &image, std::vector<Object> &objs) override;

   protected:
    void preprocess(const cv::Mat &input_image, cv::Mat &output_image) override;
    void postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) override;
    void postprocess_without_nms(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs);
    void postprocess_with_nms(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs);

   private:
    cv::Size m_input_size_ = {640, 640};
    float m_conf_thres_ = 0.25f;
    float m_nms_thres_ = 0.65f;
    int topk_ = 100;
    int strides[3] = {8, 16, 32};
    int m_grid_num_ = 8400;
    bool with_nms_ = false;
    PreParam pparam_;
};