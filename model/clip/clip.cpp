#include "model/clip/clip.h"
#include <yaml-cpp/yaml.h>

using namespace clip;

static void normalize(IOTensor& tensor, size_t size) {
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

static void ReadPrompt(const std::string& prompt_path, std::vector<std::string>& prompts) {
    std::ifstream file(prompt_path);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            prompts.push_back(line); // 逐行读取文件内容并存储到 vector 中
        }
        file.close(); // 关闭文件
    } else {
        std::cout << "无法打开文件" << std::endl;
    }
}

static void ReadTextEmbedding(const std::string& path, std::vector<float>& text_embeddings) {
    std::streampos size;
    std::ifstream fin(path.c_str(), std::ios::binary | std::ios::in);
    fin.seekg(0, std::ios::end);
    size = fin.tellg();
    text_embeddings.resize(size/sizeof(float));
    fin.seekg(0, std::ios::beg);
    fin.read((char *)text_embeddings.data(), size);
    fin.close();
}

Clip::Clip(const std::string& image_encoder_cfg, const std::string& text_encoder_cfg) {
    m_image_encoder_ = std::make_shared<ImageEncoder>(image_encoder_cfg);

    YAML::Node yaml_node = YAML::LoadFile(text_encoder_cfg);
    bool online = yaml_node["online"].as<bool>();
    if (online) {
        m_text_encoder_ = std::make_shared<TextEncoder>(text_encoder_cfg);
    } 

    std::string prompt_path = yaml_node["prompts"].as<std::string>();
    std::vector<std::string> prompts;
    ReadPrompt(prompt_path, prompts);

    std::string text_embedding_path = yaml_node["text_embedding"].as<std::string>();
    std::vector<float> embeddings;
    ReadTextEmbedding(text_embedding_path, embeddings);

    float* ptr = embeddings.data();
    for (size_t i = 0; i < prompts.size(); i++) {
        cache_[prompts[i]] = std::vector<float>(512, 0.0);
        for (size_t j = 0; j < 512; j++) {
            cache_[prompts[i]][j] = *ptr++;
        }
    }
}

void Clip::encodeImages(const std::vector<cv::Mat>& images) {
    m_image_encoder_->forward(images, image_embeddings);
    normalize(image_embeddings, images.size());
}

void Clip::encodeTexts(const std::vector<std::string>& texts) {
    std::vector<std::string> texts_not_in_cache;
    for (const auto& text: texts) {
        if (!cache_.count(text)) {
            texts_not_in_cache.push_back(text);
        }
    }

    float* ptr;

    if (!texts_not_in_cache.empty()) {
        if (!m_text_encoder_) {
            std::cout << "The text encoder is offline. Failed to generate text embeddings for text out of prompt list" << std::endl;
            exit(0);
        }
        IOTensor embeddings;
        m_text_encoder_->forward(texts_not_in_cache, embeddings);
        ptr = (float*)embeddings.data();
        for (size_t i = 0; i < texts_not_in_cache.size(); i++) {
            cache_[texts_not_in_cache[i]] = std::vector<float>(512, 0.0);
            for (size_t j = 0; j < 512; j++) {
                cache_[texts_not_in_cache[i]][j] = *ptr++;
            }
        }
    }

    text_embeddings.resize(texts.size() * 512 * sizeof(float));
    text_embeddings.shape = std::vector<int64_t>{static_cast<int64_t>(texts.size()), 512};
    text_embeddings.data_type = DataType::FP32;

    ptr = (float*)text_embeddings.data();
    for (const auto& text : texts) {
        memcpy(ptr, cache_[text].data(), 512 * sizeof(float));
        ptr += 512;
    }
    
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