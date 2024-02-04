#include "framework/framework.h"
#include "common/common.h"

#include "model/crnn/crnn.h"
#include <dirent.h>
#include <sys/stat.h>

void CrnnTest() {
    std::string current_path = "../";
    std::string yaml_file = current_path + "config/crnn/crnn.yaml";

    Crnn model(yaml_file);

    std::vector<std::string> imagePathList;
    std::string input_path = current_path + "test/image/ocr"; 
    std::string output_path = current_path + "output/crnn";
    cv::glob(input_path + "/*.png", imagePathList);

    cv::Mat image, res;

    for (auto& path : imagePathList) {
        image = cv::imread(path);
        std::string output = model.detect(image);
        std::cout << path << ": " << output << std::endl;
    }
}

int main() {
    CrnnTest();
}