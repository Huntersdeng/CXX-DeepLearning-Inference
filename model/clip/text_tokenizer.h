#pragma once
#include <zlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <vector>

namespace clip {

class TextTokenizer {
   public:
    TextTokenizer() = delete;
    TextTokenizer(const std::string &path);

    std::vector<int> tokenize(const std::string &text, size_t context_length = 77);
    std::vector<std::vector<int>> batchTokenize(const std::vector<std::string> &texts, size_t context_length = 77);

   private:
    std::string bpe(const std::string &token);
    void encode(const std::string &str, std::vector<int> &bpe_tokens);
    std::string decode(const std::vector<int> &bpe_tokens);

    std::map<size_t, std::string> byte_encoder;
    std::map<std::string, size_t> byte_decoder;
    std::map<std::string, size_t> encoder;
    std::map<size_t, std::string> decoder;
    std::map<std::vector<std::string>, size_t> bpe_ranks;
    std::map<std::string, std::string> cache;
    std::regex pattern;
};
}  // namespace clip
