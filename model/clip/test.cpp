#include "model/clip/text_tokenizer.h"
#include <unicode/ustream.h>
#include <unicode/unistr.h>

int main() {
//   Tokenizer tokenizer(Tokenizer::Mode::Conservative, Tokenizer::Flags::JoinerAnnotate);
//   std::vector<std::string> tokens;
//   tokenizer.tokenize("a photo of a man", tokens);
//   for (const auto& token : tokens) {
//     std::cout << token << std::endl;
//   }
    TextTokenizer tokenizer("/home/stardust/my_work/model-zoo-cxx/weights/clip/bpe_simple_vocab_16e6.txt.gz");
    std::vector<int> tokens = tokenizer.tokenize("a photo of a woman");
    for(const auto& token : tokens) {
        std::cout << token << ",";
    }
    std::cout << std::endl;
    return 0;
}