#include "model/base/cv_model.h"

class ImageEncoder : public CvModel {
   public:
    ImageEncoder() = delete;
    ImageEncoder(const std::string &model_path, const std::string framework_type);
    ImageEncoder(const std::string &yaml_file);
    virtual ~ImageEncoder();
    void forward(const cv::Mat &image, IOTensor& features);

    cv::Size input_size() const { return m_input_size_; }
    cv::Size output_size() const { return m_output_size_; }
   
   protected:
    void preprocess(const cv::Mat &input_image, cv::Mat &output_image);

   private:
    cv::Size m_input_size_;
    cv::Size m_output_size_;
    
};