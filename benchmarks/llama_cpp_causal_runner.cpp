#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string model;
    std::string direction;
    std::string prompt;
    std::vector<float> scales;
    int32_t layer = 41;
    int32_t alice_token = 32858;
    int32_t bob_token = 15943;
    int32_t gpu_layers = 12;
};

std::string json_escape(const std::string & value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << character;
                }
        }
    }
    return output.str();
}

std::vector<float> parse_scales(const std::string & raw) {
    std::vector<float> values;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        values.push_back(std::stof(item));
    }
    if (values.empty()) {
        throw std::runtime_error("at least one scale is required");
    }
    return values;
}

Arguments parse_arguments(int argc, char ** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + key);
        }
        const std::string value = argv[++index];
        if (key == "--model") result.model = value;
        else if (key == "--direction") result.direction = value;
        else if (key == "--prompt") result.prompt = value;
        else if (key == "--scales") result.scales = parse_scales(value);
        else if (key == "--layer") result.layer = std::stoi(value);
        else if (key == "--alice-token") result.alice_token = std::stoi(value);
        else if (key == "--bob-token") result.bob_token = std::stoi(value);
        else if (key == "--gpu-layers") result.gpu_layers = std::stoi(value);
        else throw std::runtime_error("unknown argument " + key);
    }
    if (result.model.empty() || result.direction.empty() || result.prompt.empty() || result.scales.empty()) {
        throw std::runtime_error("model, direction, prompt, and scales are required");
    }
    return result;
}

std::vector<float> read_direction(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("unable to open direction file");
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("direction file is not an array of float32 values");
    }
    input.seekg(0);
    std::vector<float> result(static_cast<size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char *>(result.data()), bytes);
    if (!input) throw std::runtime_error("unable to read direction file");
    return result;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(256);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(static_cast<size_t>(-count));
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    if (count < 0) throw std::runtime_error("unable to decode token piece");
    return std::string(buffer.data(), static_cast<size_t>(count));
}

double logsumexp(const float * logits, int32_t count) {
    const float maximum = *std::max_element(logits, logits + count);
    double total = 0.0;
    for (int32_t index = 0; index < count; ++index) {
        total += std::exp(static_cast<double>(logits[index] - maximum));
    }
    return static_cast<double>(maximum) + std::log(total);
}

double kl_from_base(
    const std::vector<float> & logits,
    const std::vector<float> & baseline
) {
    if (baseline.empty()) return 0.0;
    const double current_norm = logsumexp(logits.data(), logits.size());
    const double base_norm = logsumexp(baseline.data(), baseline.size());
    double result = 0.0;
    for (size_t index = 0; index < logits.size(); ++index) {
        const double log_probability = logits[index] - current_norm;
        const double base_log_probability = baseline[index] - base_norm;
        const double probability = std::exp(log_probability);
        result += probability * (log_probability - base_log_probability);
    }
    return result;
}

void quiet_log(enum ggml_log_level, const char *, void *) {}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        llama_log_set(quiet_log, nullptr);
        ggml_backend_load_all();

        llama_model_params model_parameters = llama_model_default_params();
        model_parameters.n_gpu_layers = arguments.gpu_layers;
        llama_model * model = llama_model_load_from_file(arguments.model.c_str(), model_parameters);
        if (model == nullptr) throw std::runtime_error("unable to load GGUF model");

        const int32_t hidden_width = llama_model_n_embd(model);
        const int32_t layer_count = llama_model_n_layer(model);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
        std::vector<float> direction = read_direction(arguments.direction);
        if (static_cast<int32_t>(direction.size()) != hidden_width) {
            throw std::runtime_error("direction width does not match model hidden width");
        }
        if (arguments.layer <= 0 || arguments.layer >= layer_count) {
            throw std::runtime_error("attachment layer is outside the supported control-vector range");
        }

        const int32_t token_count = -llama_tokenize(
            vocab, arguments.prompt.c_str(), arguments.prompt.size(), nullptr, 0, true, true
        );
        std::vector<llama_token> tokens(token_count);
        if (llama_tokenize(
                vocab, arguments.prompt.c_str(), arguments.prompt.size(), tokens.data(),
                tokens.size(), true, true
            ) < 0) {
            throw std::runtime_error("unable to tokenize prompt");
        }

        llama_context_params context_parameters = llama_context_default_params();
        context_parameters.n_ctx = 512;
        context_parameters.n_batch = 512;
        context_parameters.n_ubatch = 512;
        context_parameters.n_threads = 8;
        context_parameters.n_threads_batch = 8;
        context_parameters.no_perf = true;
        llama_context * context = llama_init_from_model(model, context_parameters);
        if (context == nullptr) throw std::runtime_error("unable to create llama context");

        std::vector<float> baseline;
        std::cout << "{\"model_hidden_width\":" << hidden_width
                  << ",\"model_layer_count\":" << layer_count
                  << ",\"vocabulary_size\":" << vocabulary_size
                  << ",\"prompt_token_count\":" << token_count
                  << ",\"conditions\":[";

        for (size_t condition = 0; condition < arguments.scales.size(); ++condition) {
            const float scale = arguments.scales[condition];
            llama_memory_clear(llama_get_memory(context), true);
            if (scale == 0.0f) {
                if (llama_set_adapter_cvec(
                        context, nullptr, 0, hidden_width, arguments.layer, arguments.layer
                    ) != 0) {
                    throw std::runtime_error("unable to disable control vector");
                }
            } else {
                std::vector<float> layer_data(
                    static_cast<size_t>(hidden_width) * arguments.layer, 0.0f
                );
                const size_t offset = static_cast<size_t>(hidden_width) * (arguments.layer - 1);
                for (int32_t index = 0; index < hidden_width; ++index) {
                    layer_data[offset + index] = direction[index] * scale;
                }
                if (llama_set_adapter_cvec(
                        context, layer_data.data(), layer_data.size(), hidden_width,
                        arguments.layer, arguments.layer
                    ) != 0) {
                    throw std::runtime_error("unable to apply control vector");
                }
            }

            const auto started = std::chrono::steady_clock::now();
            llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("llama decode failed");
            }
            float * raw_logits = llama_get_logits_ith(context, -1);
            if (raw_logits == nullptr) throw std::runtime_error("llama logits unavailable");
            std::vector<float> logits(raw_logits, raw_logits + vocabulary_size);
            const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started
            ).count();
            const int32_t generated = static_cast<int32_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin()
            );
            const double normalization = logsumexp(logits.data(), vocabulary_size);
            double maximum_difference = 0.0;
            if (baseline.empty()) {
                baseline = logits;
            } else {
                for (size_t index = 0; index < logits.size(); ++index) {
                    maximum_difference = std::max(
                        maximum_difference,
                        std::abs(static_cast<double>(logits[index] - baseline[index]))
                    );
                }
            }
            if (condition != 0) std::cout << ',';
            std::cout << "{\"scale\":" << std::setprecision(9) << scale
                      << ",\"alice_logit\":" << logits.at(arguments.alice_token)
                      << ",\"bob_logit\":" << logits.at(arguments.bob_token)
                      << ",\"alice_probability\":"
                      << std::exp(logits.at(arguments.alice_token) - normalization)
                      << ",\"bob_probability\":"
                      << std::exp(logits.at(arguments.bob_token) - normalization)
                      << ",\"generated_token_id\":" << generated
                      << ",\"generated\":\"" << json_escape(token_piece(vocab, generated)) << "\""
                      << ",\"kl_from_base\":" << kl_from_base(logits, baseline)
                      << ",\"max_abs_logit_difference_from_base\":" << maximum_difference
                      << ",\"latency_ms\":" << elapsed << '}';
        }
        std::cout << "]}\n";

        llama_free(context);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
