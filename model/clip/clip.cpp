#include "model/clip/clip.h"

using namespace clip;

void normalize(IOTensor& tensor, size_t size) {
    float *ptr = (float*)tensor.data();
    for (size_t i = 0; i < size; i++) {
        float norm = 0.0;
        for (size_t j = 0; j < 512; j++) {
            norm += std::pow(*(ptr+j), 2);
        }
        norm = std::sqrt(norm);

        for (size_t j = 0; j < 512; j++) {
            *ptr = *ptr / norm;
            ++ptr;
        }
    }
}

Clip::Clip(const std::string& image_encoder_cfg, const std::string& text_encoder_cfg) {
    m_image_encoder_ = std::make_shared<ImageEncoder>(image_encoder_cfg);
    m_text_encoder_ = std::make_shared<TextEncoder>(text_encoder_cfg);
}

void Clip::encodeImages(const std::vector<cv::Mat>& images) {
    m_image_encoder_->forward(images, image_embeddings);
    normalize(image_embeddings, images.size());
}

void Clip::encodeTexts(const std::vector<std::string>& texts) {
    m_text_encoder_->forward(texts, text_embeddings);
    normalize(text_embeddings, texts.size());
}

std::vector<std::vector<float>> Clip::computeProbabilities() {
    size_t num_images = image_embeddings.shape[0];
    size_t num_texts = text_embeddings.shape[0];
    cv::Mat image_matrix(num_images, 512, CV_32F, image_embeddings.data());
    cv::Mat text_matrix(num_texts, 512, CV_32F, text_embeddings.data());
    cv::Mat logits;
    cv::gemm(image_matrix, text_matrix.t(), 100, cv::Mat(), 0.0, logits);
    
    std::vector<std::vector<float>> probs;
    float *ptr = logits.ptr<float>();
    for (size_t i = 0; i < num_images; i++) {
        float exp_sum = 0.0;
        for (size_t j = 0; j < num_texts; j++) {
            exp_sum += std::exp(*(ptr+j));
        }
        std::vector<float> prob;
        for (size_t j = 0; j < num_texts; j++) {
            prob.push_back(std::exp(*(ptr++)) / exp_sum);
        }
        probs.push_back(prob);
    }
    return probs;
}