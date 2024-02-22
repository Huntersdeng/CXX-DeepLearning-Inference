#include "model/clip/text_tokenizer.h"
#include "model/clip/image_encoder.h"
#include "model/clip/text_encoder.h"
#include "model/clip/clip.h"

void ModuleTest() {
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
}

void GetTextEmbeddings() {
    std::string current_path = "../";
    std::string text_encoder_cfg = current_path + "config/clip/text_encoder.yaml";
    clip::TextEncoder text_encoder(text_encoder_cfg);

    std::string prompt_path = current_path + "config/clip/prompts.txt";
    std::ifstream file(prompt_path);
    std::vector<std::string> texts;

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            texts.push_back(line); // 逐行读取文件内容并存储到 vector 中
        }
        file.close(); // 关闭文件
    } else {
        std::cout << "无法打开文件" << std::endl;
    }

    std::cout << "Prompts: ";
    for (const auto& l : texts) {
        std::cout << l << ", ";
    }
    std::cout << std::endl;

    IOTensor text_embeddings;
    text_encoder.forward(texts, text_embeddings);
    std::cout << "Shape of image text_embeddings: [";
    for (int64_t i : text_embeddings.shape) {
        std::cout << i << ",";
    }
    std::cout << "]" << std::endl;

    
    std::string fname = current_path + "weights/clip/text_embeddings.bin";
    std::ofstream fout(fname.c_str(), std::ios::binary | std::ios::out);
    fout.write((char *)text_embeddings.data(), text_embeddings.size());
    fout.close();
}

void PipeLineTest() {
    std::string current_path = "../";
    std::string image_encoder_cfg = current_path + "config/clip/image_encoder.yaml";
    std::string text_encoder_cfg = current_path + "config/clip/text_encoder.yaml";

    clip::Clip clip_model(image_encoder_cfg, text_encoder_cfg);

    std::vector<cv::Mat> images;
    images.push_back(cv::imread("../test/image/clip/franz-kafka.jpg"));
    images.push_back(cv::imread("../test/image/clip/Mona_Lisa.jpg"));
    clip_model.encodeImages(images);

    std::vector<std::string> texts{"a photo of a man", "a photo of a woman"};
    clip_model.encodeTexts(texts);

    std::vector<std::vector<float>> probs = clip_model.computeProbabilities();

    std::cout << "[ ";
    for (size_t i = 0; i < probs.size(); i++) {
        std::cout << "[ ";
        for (size_t j = 0; j < probs[0].size(); j++) {
            std::cout << probs[i][j] << " ";
        }
        std::cout << " ], ";
    }
    std::cout << " ]" << std::endl;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "-g") {
        GetTextEmbeddings();
    }
    PipeLineTest();
}