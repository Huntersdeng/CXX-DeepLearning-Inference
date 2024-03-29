#pragma once
#include "model/base/model.h"

namespace sam {
class MaskDecoder : public Model {
   public:
    MaskDecoder() = delete;
    MaskDecoder(const std::string &yaml_file);
    virtual ~MaskDecoder();
    void forward(const IOTensor &features, const std::vector<cv::Point2f> &image_point_coords,
                 const std::vector<float> &image_point_labels, cv::Mat &low_res_mask);

   private:
    std::vector<int64_t> features_shape;
};
}