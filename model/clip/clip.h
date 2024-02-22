#pragma once
#include "model/clip/image_encoder.h"
#include "model/clip/text_encoder.h"

namespace clip {

class Clip {
   public:
    Clip() = delete;
    Clip(const std::string& image_encoder_cfg, const std::string& text_encoder_cfg);
    void encodeImages(const std::vector<cv::Mat>& images);
    void encodeTexts(const std::vector<std::string>& texts);
    std::vector<std::vector<float>> computeProbabilities();
   private:
    std::shared_ptr<ImageEncoder> m_image_encoder_;
    std::shared_ptr<TextEncoder> m_text_encoder_;
    IOTensor image_embeddings;
    IOTensor text_embeddings;
};
}