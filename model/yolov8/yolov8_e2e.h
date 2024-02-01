#pragma once
#include <fstream>

#include "model/base/cv_model.h"

class YOLOv8E2E : public CvModel {
   public:
    YOLOv8E2E() = delete;
    explicit YOLOv8E2E(const std::string &model_path, const std::string framework_type, cv::Size input_size, int topk);
    explicit YOLOv8E2E(const std::string &yaml_file);
    ~YOLOv8E2E();

    void detect(const cv::Mat &image, std::vector<Object> &objs) override;

   protected:
    void postprocess(const std::unordered_map<std::string, IOTensor> &output, std::vector<Object> &objs) override;

   private:
    cv::Size m_input_size_ = {640, 640};
    int topk_ = 100;
    PreParam pparam_;
};