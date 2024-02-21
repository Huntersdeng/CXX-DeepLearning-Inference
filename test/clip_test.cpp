#include "model/clip/text_tokenizer.h"
#include "model/clip/image_encoder.h"
#include "model/clip/text_encoder.h"

int main() {
//   Tokenizer tokenizer(Tokenizer::Mode::Conservative, Tokenizer::Flags::JoinerAnnotate);
//   std::vector<std::string> tokens;
//   tokenizer.tokenize("a photo of a man", tokens);
//   for (const auto& token : tokens) {
//     std::cout << token << std::endl;
//   }
    clip::TextTokenizer tokenizer("/home/stardust/my_work/model-zoo-cxx/weights/clip/bpe_simple_vocab_16e6.txt.gz");
    std::vector<int> tokens = tokenizer.tokenize("a photo of a woman");
    for(int token : tokens) {
        std::cout << token << ",";
    }
    std::cout << std::endl;

    std::string current_path = "../";
    std::string image_encoder_cfg = current_path + "config/clip/image_encoder.yaml";
    std::string text_encoder_cfg = current_path + "config/clip/text_encoder.yaml";

    clip::ImageEncoder image_encoder(image_encoder_cfg);
    
    std::vector<cv::Mat> images;
    images.push_back(cv::imread("../test/image/clip/franz-kafka.jpg"));

    IOTensor image_embeddings;
    image_encoder.forward(images, image_embeddings);
    std::cout << "Shape of image image_embeddings: [";
    for (int64_t i : image_embeddings.shape) {
        std::cout << i << ",";
    }
    std::cout << "]" << std::endl;

    float* ptr = (float*)image_embeddings.data();
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    for (size_t i = 0; i < image_embeddings.size() / 4; i++) {
        float val = *ptr++;
        if (val > max_val) {
            max_val = val;
        }
        if (val < min_val) {
            min_val = val;
        }
    }
    std::cout << "Range of image_embeddings: [" << min_val << "," << max_val << "]" << std::endl;

    clip::TextEncoder text_encoder(text_encoder_cfg);
    
    std::vector<std::string> texts{"a photo of a man", "a photo of a woman"};

    IOTensor text_embeddings;
    text_encoder.forward(texts, text_embeddings);
    std::cout << "Shape of image text_embeddings: [";
    for (int64_t i : text_embeddings.shape) {
        std::cout << i << ",";
    }
    std::cout << "]" << std::endl;

    ptr = (float*)text_embeddings.data();
    min_val = FLT_MAX;
    max_val = -FLT_MAX;
    for (size_t i = 0; i < text_embeddings.size() / 4; i++) {
        float val = *ptr++;
        if (val > max_val) {
            max_val = val;
        }
        if (val < min_val) {
            min_val = val;
        }
    }
    std::cout << "Range of text_embeddings: [" << min_val << "," << max_val << "]" << std::endl;

    std::vector<float> norm_image_embeddings;

    ptr = (float*)image_embeddings.data();
    for (size_t i = 0; i < images.size(); i++) {
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

    ptr = (float*)text_embeddings.data();
    for (size_t i = 0; i < texts.size(); i++) {
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

    ptr = (float*)image_embeddings.data();
    min_val = FLT_MAX;
    max_val = -FLT_MAX;
    for (size_t i = 0; i < image_embeddings.size() / 4; i++) {
        float val = *ptr++;
        if (val > max_val) {
            max_val = val;
        }
        if (val < min_val) {
            min_val = val;
        }
    }
    std::cout << "After normalization, range of image_embeddings: [" << min_val << "," << max_val << "]" << std::endl;

    ptr = (float*)text_embeddings.data();
    min_val = FLT_MAX;
    max_val = -FLT_MAX;
    for (size_t i = 0; i < text_embeddings.size() / 4; i++) {
        float val = *ptr++;
        if (val > max_val) {
            max_val = val;
        }
        if (val < min_val) {
            min_val = val;
        }
    }
    std::cout << "After normalization, range of text_embeddings: [" << min_val << "," << max_val << "]" << std::endl;

    cv::Mat image_matrix(images.size(), 512, CV_32F, image_embeddings.data());
    cv::Mat text_matrix(texts.size(), 512, CV_32F, text_embeddings.data());
    cv::Mat result;
    cv::gemm(image_matrix, text_matrix.t(), 100, cv::Mat(), 0.0, result);
    std::cout << result << std::endl;
    
    return 0;
}