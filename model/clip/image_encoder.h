#pragma once
#include "model/base/model.h"

namespace clip {

class ImageEncoder : public Model {
   public:
    ImageEncoder() = delete;
    ImageEncoder(const std::string &model_path, const std::string framework_type);
    ImageEncoder(const std::string &yaml_file);
    virtual ~ImageEncoder();
    void forward(const std::vector<cv::Mat> &images, IOTensor &features);

    cv::Size input_size() const { return m_input_size_; }
    size_t output_size() const { return m_output_size_; }

   protected:
    void preprocess(const cv::Mat &input_image, cv::Mat &output_image);

   private:
    cv::Size m_input_size_;
    size_t m_output_size_;
};

}