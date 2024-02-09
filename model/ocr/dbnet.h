#pragma once
#include <fstream>

#include "model/base/detection_model.h"

class DBNet : public DetectionModel {
   public:
    DBNet() = delete;
    explicit DBNet(const std::string &model_path, const std::string framework_type, cv::Size input_size, float box_thres);
    explicit DBNet(const std::string &yaml_file);
    ~DBNet();

    void detect(const cv::Mat &image, std::vector<Object> &objs) override;

   protected:
    void preprocess(const cv::Mat &input_image, cv::Mat &output_image) override;
    void postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) override;

   private:
    cv::Size m_input_size_ = {640, 640};
    float m_box_thres_ = 0.3f;
    float m_expand_ratio_ = 1.5f;
    float m_score_thres_ = 0.3f;
    int m_box_min_size_ = 5;
    PreParam pparam_;
};