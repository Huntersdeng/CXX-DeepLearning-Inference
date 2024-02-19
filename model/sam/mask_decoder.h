#include "model/base/cv_model.h"

class MaskDecoder : public CvModel {
   public:
    MaskDecoder() = delete;
    MaskDecoder(const std::string &model_path, const std::string framework_type);
    MaskDecoder(const std::string &yaml_file);
    virtual ~MaskDecoder();
    void forward(const IOTensor &features, const std::vector<cv::Point2f> &image_point_coords,
                 const std::vector<float> &image_point_labels, cv::Mat &low_res_mask);

   private:
    std::vector<int64_t> features_shape;
};