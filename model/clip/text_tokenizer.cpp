#include "model/clip/text_tokenizer.h"

using namespace clip;

static std::map<size_t, std::string> ByteToUnicode() {
    return {{33, "!"},  {34, "\""}, {35, "#"},  {36, "$"},  {37, "%"},  {38, "&"},  {39, "'"},  {40, "("},  {41, ")"},
            {42, "*"},  {43, "+"},  {44, ","},  {45, "-"},  {46, "."},  {47, "/"},  {48, "0"},  {49, "1"},  {50, "2"},
            {51, "3"},  {52, "4"},  {53, "5"},  {54, "6"},  {55, "7"},  {56, "8"},  {57, "9"},  {58, ":"},  {59, ";"},
            {60, "<"},  {61, "="},  {62, ">"},  {63, "?"},  {64, "@"},  {65, "A"},  {66, "B"},  {67, "C"},  {68, "D"},
            {69, "E"},  {70, "F"},  {71, "G"},  {72, "H"},  {73, "I"},  {74, "J"},  {75, "K"},  {76, "L"},  {77, "M"},
            {78, "N"},  {79, "O"},  {80, "P"},  {81, "Q"},  {82, "R"},  {83, "S"},  {84, "T"},  {85, "U"},  {86, "V"},
            {87, "W"},  {88, "X"},  {89, "Y"},  {90, "Z"},  {91, "["},  {92, "\\"}, {93, "]"},  {94, "^"},  {95, "_"},
            {96, "`"},  {97, "a"},  {98, "b"},  {99, "c"},  {100, "d"}, {101, "e"}, {102, "f"}, {103, "g"}, {104, "h"},
            {105, "i"}, {106, "j"}, {107, "k"}, {108, "l"}, {109, "m"}, {110, "n"}, {111, "o"}, {112, "p"}, {113, "q"},
            {114, "r"}, {115, "s"}, {116, "t"}, {117, "u"}, {118, "v"}, {119, "w"}, {120, "x"}, {121, "y"}, {122, "z"},
            {123, "{"}, {124, "|"}, {125, "}"}, {126, "~"}, {161, "¡"}, {162, "¢"}, {163, "£"}, {164, "¤"}, {165, "¥"},
            {166, "¦"}, {167, "§"}, {168, "¨"}, {169, "©"}, {170, "ª"}, {171, "«"}, {172, "¬"}, {174, "®"}, {175, "¯"},
            {176, "°"}, {177, "±"}, {178, "²"}, {179, "³"}, {180, "´"}, {181, "µ"}, {182, "¶"}, {183, "·"}, {184, "¸"},
            {185, "¹"}, {186, "º"}, {187, "»"}, {188, "¼"}, {189, "½"}, {190, "¾"}, {191, "¿"}, {192, "À"}, {193, "Á"},
            {194, "Â"}, {195, "Ã"}, {196, "Ä"}, {197, "Å"}, {198, "Æ"}, {199, "Ç"}, {200, "È"}, {201, "É"}, {202, "Ê"},
            {203, "Ë"}, {204, "Ì"}, {205, "Í"}, {206, "Î"}, {207, "Ï"}, {208, "Ð"}, {209, "Ñ"}, {210, "Ò"}, {211, "Ó"},
            {212, "Ô"}, {213, "Õ"}, {214, "Ö"}, {215, "×"}, {216, "Ø"}, {217, "Ù"}, {218, "Ú"}, {219, "Û"}, {220, "Ü"},
            {221, "Ý"}, {222, "Þ"}, {223, "ß"}, {224, "à"}, {225, "á"}, {226, "â"}, {227, "ã"}, {228, "ä"}, {229, "å"},
            {230, "æ"}, {231, "ç"}, {232, "è"}, {233, "é"}, {234, "ê"}, {235, "ë"}, {236, "ì"}, {237, "í"}, {238, "î"},
            {239, "ï"}, {240, "ð"}, {241, "ñ"}, {242, "ò"}, {243, "ó"}, {244, "ô"}, {245, "õ"}, {246, "ö"}, {247, "÷"},
            {248, "ø"}, {249, "ù"}, {250, "ú"}, {251, "û"}, {252, "ü"}, {253, "ý"}, {254, "þ"}, {255, "ÿ"}, {0, "Ā"},
            {1, "ā"},   {2, "Ă"},   {3, "ă"},   {4, "Ą"},   {5, "ą"},   {6, "Ć"},   {7, "ć"},   {8, "Ĉ"},   {9, "ĉ"},
            {10, "Ċ"},  {11, "ċ"},  {12, "Č"},  {13, "č"},  {14, "Ď"},  {15, "ď"},  {16, "Đ"},  {17, "đ"},  {18, "Ē"},
            {19, "ē"},  {20, "Ĕ"},  {21, "ĕ"},  {22, "Ė"},  {23, "ė"},  {24, "Ę"},  {25, "ę"},  {26, "Ě"},  {27, "ě"},
            {28, "Ĝ"},  {29, "ĝ"},  {30, "Ğ"},  {31, "ğ"},  {32, "Ġ"},  {127, "ġ"}, {128, "Ģ"}, {129, "ģ"}, {130, "Ĥ"},
            {131, "ĥ"}, {132, "Ħ"}, {133, "ħ"}, {134, "Ĩ"}, {135, "ĩ"}, {136, "Ī"}, {137, "ī"}, {138, "Ĭ"}, {139, "ĭ"},
            {140, "Į"}, {141, "į"}, {142, "İ"}, {143, "ı"}, {144, "Ĳ"}, {145, "ĳ"}, {146, "Ĵ"}, {147, "ĵ"}, {148, "Ķ"},
            {149, "ķ"}, {150, "ĸ"}, {151, "Ĺ"}, {152, "ĺ"}, {153, "Ļ"}, {154, "ļ"}, {155, "Ľ"}, {156, "ľ"}, {157, "Ŀ"},
            {158, "ŀ"}, {159, "Ł"}, {160, "ł"}, {173, "Ń"}};
}

static std::vector<std::string> Unicode() {
    return {"!", "\"", "#", "$", "%", "&", "\'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4",
            "5", "6",  "7", "8", "9", ":", ";",  "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H",
            "I", "J",  "K", "L", "M", "N", "O",  "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\",
            "]", "^",  "_", "`", "a", "b", "c",  "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
            "q", "r",  "s", "t", "u", "v", "w",  "x", "y", "z", "{", "|", "}", "~", "¡", "¢", "£", "¤", "¥", "¦",
            "§", "¨",  "©", "ª", "«", "¬", "®",  "¯", "°", "±", "²", "³", "´", "µ", "¶", "·", "¸", "¹", "º", "»",
            "¼", "½",  "¾", "¿", "À", "Á", "Â",  "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï",
            "Ð", "Ñ",  "Ò", "Ó", "Ô", "Õ", "Ö",  "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý", "Þ", "ß", "à", "á", "â", "ã",
            "ä", "å",  "æ", "ç", "è", "é", "ê",  "ë", "ì", "í", "î", "ï", "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "÷",
            "ø", "ù",  "ú", "û", "ü", "ý", "þ",  "ÿ", "Ā", "ā", "Ă", "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ",
            "Č", "č",  "Ď", "ď", "Đ", "đ", "Ē",  "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě", "Ĝ", "ĝ", "Ğ", "ğ",
            "Ġ", "ġ",  "Ģ", "ģ", "Ĥ", "ĥ", "Ħ",  "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į", "į", "İ", "ı", "Ĳ", "ĳ",
            "Ĵ", "ĵ",  "Ķ", "ķ", "ĸ", "Ĺ", "ĺ",  "Ļ", "ļ", "Ľ", "ľ", "Ŀ", "ŀ", "Ł", "ł", "Ń"};
}

static void lower(std::string &str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
}

static void whitespaceClean(std::string &str) {
    str = std::regex_replace(str, std::regex("\\s+"), " ");
    str = str.substr(str.find_first_not_of(" "), str.find_last_not_of(" ") + 1);
}

static void getPairs(const std::vector<std::string> &word, std::set<std::vector<std::string>> &pairs) {
    pairs.clear();
    std::string prev_char = word[0];
    for (size_t i = 1; i < word.size(); ++i) {
        pairs.insert(std::vector<std::string>{prev_char, word[i]});
        prev_char = word[i];
    }
}

static void ReadVocab(const std::string& path, std::vector<std::vector<std::string>>& merges) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (file == NULL) {
        std::cerr << "Error opening file" << std::endl;
        exit(0);
    }

    std::stringstream ss;
    char buffer[1024];
    int num_read = 0;
    while ((num_read = gzread(file, buffer, 1024)) > 0) {
        ss.write(buffer, num_read);
    }

    gzclose(file);

    std::string content = ss.str();

    // Assuming content is UTF-8 encoded, you can split it by lines
    std::istringstream content_stream(content);
    std::string line;
    size_t cnt = 0;
    while (std::getline(content_stream, line, '\n')) {
        if (cnt > 0 && cnt < 49152 - 256 - 2 + 1) {
            ss.str("");
            ss.clear();
            ss << line;
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ' ')) {
                tokens.push_back(token);
            }
            merges.push_back(tokens);
        }
        cnt++;
    }
}

TextTokenizer::TextTokenizer(const std::string& path)
    : pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)") {
    byte_encoder = ByteToUnicode();

    for (const auto& kv : byte_encoder) {
        byte_decoder[kv.second] = kv.first;
    }

    std::vector<std::vector<std::string>> merges;
    ReadVocab(path, merges);

    std::vector<std::string> vocab = Unicode();
    size_t size = vocab.size();
    for (size_t i = 0; i < size; i++) {
        vocab.push_back(vocab[i] + "</w>");
    }

    for (const auto& merge : merges) {
        std::string str;
        for (const auto& s : merge) {
            str += s;
        }
        vocab.push_back(str);
    }
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");

    for (size_t i = 0; i < vocab.size(); i++) {
        encoder[vocab[i]] = i;
        decoder[i] = vocab[i];
    }

    for (size_t i = 0; i < merges.size(); i++) {
        bpe_ranks[merges[i]] = i;
    }
}

std::string TextTokenizer::bpe(const std::string &token) {
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }

    std::vector<std::string> word;
    for (size_t i = 0; i < token.size(); i++) {
        if (i < token.size() - 1) {
            word.push_back(std::string(1, token[i]));
        } else {
            word.push_back(std::string(1, token[i]) + "</w>");
        }
    }

    std::set<std::vector<std::string>> pairs;
    getPairs(word, pairs);
    if (pairs.empty()) {
        return token + "</w>";
    }

    while(true) {
        int idx = INT_MAX;
        std::vector<std::string> bigram{"", ""};
        for (const auto& pair : pairs) {
            if (bpe_ranks.find(pair) != bpe_ranks.end()) {
                int cur_idx = bpe_ranks[pair];
                if (cur_idx < idx) {
                    idx = cur_idx;
                    bigram = pair;
                }
            }
        }
        if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
            break;
        }
        std::string first = bigram[0], second = bigram[1];
        std::vector<std::string> new_word;
        auto i = word.begin();
        while (i < word.end()) {
            auto j = std::find(i, word.end(), first);
            if (j != word.end()) {
                for (auto x = i; x < j; x++) {
                    new_word.push_back(*x);
                }
                i = j;
            } else {
                for (auto x = i; x < word.end(); x++) {
                    new_word.push_back(*x);
                }
                break;
            }
            if (*i == first && i < word.end() - 1 && *(i+1) == second) {
                new_word.push_back(first + second);
                i = i + 2;
            } else {
                new_word.push_back(*i);
                i = i + 1;
            }
        }
        word = new_word;
        if (word.size() == 1) {
            break;
        } else {
            getPairs(word, pairs);
        }
    }
    std::string word_str;
    for (const auto& w : word) {
        word_str += w;
    }
    cache[token] = word_str;
    return word_str;
}

void TextTokenizer::encode(const std::string &str, std::vector<int>& bpe_tokens) {
    bpe_tokens.clear();
    std::string text = str;
    whitespaceClean(text);
    lower(text);

    std::smatch m;
    std::vector<std::string> words;
    while (std::regex_search(text, m, pattern)) {
        for (auto x : m) {
            if (x.str().find(" ") == 0) {
                words.push_back(x.str().substr(1));
            } else {
                words.push_back(x.str());
            }
            
        }
        text = m.suffix();
    }

    for (const auto& word : words) {
        std::string tokens = bpe(word);

        std::stringstream ss(tokens);
        std::string bpe_token;
        while (std::getline(ss, bpe_token, ' ')) {
            bpe_tokens.push_back(encoder[bpe_token]);
        }
    }
}

std::string TextTokenizer::decode(const std::vector<int>& tokens) {
    std::string str;
    for (int token: tokens) {
        str += decoder[token];
    }
    return str;
}

std::vector<int> TextTokenizer::tokenize(const std::string &text, size_t context_length) {
    size_t sot_token = encoder["<|startoftext|>"];
    size_t eot_token = encoder["<|endoftext|>"];
    std::vector<int> bpe_tokens;
    bpe_tokens.push_back(sot_token);

    std::vector<int> bpe_tokens_part;
    encode(text, bpe_tokens_part);

    bpe_tokens.insert(bpe_tokens.end(), bpe_tokens_part.begin(), bpe_tokens_part.end());

    if (bpe_tokens.size() < context_length - 1) {
        bpe_tokens.push_back(eot_token);
        bpe_tokens.resize(context_length, 0);
    } else {
        bpe_tokens.resize(context_length, 0);
        bpe_tokens[context_length-1] = eot_token;
    }
    return bpe_tokens;
}

std::vector<std::vector<int>> TextTokenizer::batchTokenize(const std::vector<std::string> &texts, size_t context_length) {
    std::vector<std::vector<int>> all_tokens;
    for (const auto& text: texts) {
        all_tokens.push_back(tokenize(text, context_length));
    }
    return all_tokens;
}