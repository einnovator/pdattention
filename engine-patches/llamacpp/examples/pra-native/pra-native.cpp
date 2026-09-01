#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<llama_token> tokenize(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special) {
    const int32_t count = -llama_tokenize(
        vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
    if (count <= 0) {
        return {};
    }
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(
            vocab,
            text.c_str(),
            text.size(),
            tokens.data(),
            tokens.size(),
            add_special,
            true) < 0) {
        return {};
    }
    return tokens;
}

bool decode_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_seq_id sequence,
        llama_pos position_start,
        bool output_last) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t index = 0; index < batch.n_tokens; ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = position_start + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = sequence;
        batch.logits[index] = output_last && index + 1 == batch.n_tokens;
    }
    const int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return status == 0;
}

std::vector<float> last_logits(llama_context * ctx, int32_t vocabulary_size) {
    const float * values = llama_get_logits_ith(ctx, -1);
    if (values == nullptr) {
        return {};
    }
    return std::vector<float>(values, values + vocabulary_size);
}

llama_token argmax(const std::vector<float> & values) {
    return static_cast<llama_token>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

float maximum_error(
        const std::vector<float> & left,
        const std::vector<float> & right) {
    float result = 0.0f;
    for (size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::fabs(left[index] - right[index]));
    }
    return result;
}

struct DecodeResult {
    std::vector<std::vector<float>> logits;
    std::vector<llama_token> generated;
};

struct LifecycleResult {
    float absent_isolation_error = INFINITY;
    float reuse_error = INFINITY;
    bool absent_top_equal = false;
};

LifecycleResult run_lifecycle_control(
        llama_context * ctx,
        llama_memory_t memory,
        const std::vector<llama_token> & resource_tokens,
        const std::vector<llama_token> & query_tokens,
        int32_t vocabulary_size) {
    constexpr llama_seq_id resource_sequence = 1;
    constexpr llama_seq_id first_request = 0;
    constexpr llama_seq_id second_request = 2;
    llama_memory_clear(memory, true);
    if (!decode_tokens(ctx, resource_tokens, resource_sequence, 0, false)) {
        return {};
    }
    llama_memory_seq_cp(memory, resource_sequence, first_request, 0, -1);
    if (!decode_tokens(
            ctx,
            query_tokens,
            first_request,
            static_cast<llama_pos>(resource_tokens.size()),
            true)) {
        return {};
    }
    const auto first_logits = last_logits(ctx, vocabulary_size);

    llama_memory_seq_rm(memory, first_request, -1, -1);
    if (!decode_tokens(ctx, query_tokens, second_request, 0, true)) {
        return {};
    }
    const auto resident_but_absent_logits = last_logits(ctx, vocabulary_size);
    llama_memory_seq_rm(memory, second_request, -1, -1);

    llama_memory_seq_cp(memory, resource_sequence, second_request, 0, -1);
    if (!decode_tokens(
            ctx,
            query_tokens,
            second_request,
            static_cast<llama_pos>(resource_tokens.size()),
            true)) {
        return {};
    }
    const auto reused_logits = last_logits(ctx, vocabulary_size);

    llama_memory_clear(memory, true);
    if (!decode_tokens(ctx, query_tokens, second_request, 0, true)) {
        return {};
    }
    const auto fresh_absent_logits = last_logits(ctx, vocabulary_size);
    return {
        maximum_error(resident_but_absent_logits, fresh_absent_logits),
        maximum_error(first_logits, reused_logits),
        argmax(resident_but_absent_logits) == argmax(fresh_absent_logits),
    };
}

DecodeResult run_split_condition(
        llama_context * ctx,
        llama_memory_t memory,
        const std::vector<llama_token> & resource_tokens,
        const std::vector<llama_token> & query_tokens,
        int32_t vocabulary_size,
        bool native_attach,
        int32_t decode_steps) {
    llama_memory_clear(memory, true);
    constexpr llama_seq_id resource_sequence = 1;
    constexpr llama_seq_id request_sequence = 0;
    const llama_seq_id encode_sequence = native_attach
        ? resource_sequence
        : request_sequence;
    if (!decode_tokens(ctx, resource_tokens, encode_sequence, 0, false)) {
        return {};
    }
    if (native_attach) {
        llama_memory_seq_cp(
            memory, resource_sequence, request_sequence, 0, -1);
    }
    if (!decode_tokens(
            ctx,
            query_tokens,
            request_sequence,
            static_cast<llama_pos>(resource_tokens.size()),
            true)) {
        return {};
    }

    DecodeResult result;
    for (int32_t step = 0; step < decode_steps; ++step) {
        result.logits.push_back(last_logits(ctx, vocabulary_size));
        if (result.logits.back().empty()) {
            return {};
        }
        const llama_token token = argmax(result.logits.back());
        result.generated.push_back(token);
        if (step + 1 < decode_steps && !decode_tokens(
                ctx,
                std::vector<llama_token>{token},
                request_sequence,
                static_cast<llama_pos>(
                    resource_tokens.size() + query_tokens.size() + step),
                true)) {
            return {};
        }
    }
    return result;
}

void usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s -m model.gguf [-ngl layers] [-r resource] [-q query]\n",
        program);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string resource = "The launch code is CERULEAN-7.\n";
    std::string query = "The launch code is";
    int32_t gpu_layers = 0;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "-m") == 0 && index + 1 < argc) {
            model_path = argv[++index];
        } else if (std::strcmp(argv[index], "-ngl") == 0 && index + 1 < argc) {
            gpu_layers = std::stoi(argv[++index]);
        } else if (std::strcmp(argv[index], "-r") == 0 && index + 1 < argc) {
            resource = argv[++index];
        } else if (std::strcmp(argv[index], "-q") == 0 && index + 1 < argc) {
            query = argv[++index];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (model_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto resource_tokens = tokenize(vocab, resource, true);
    const auto query_tokens = tokenize(vocab, query, false);
    if (resource_tokens.empty() || query_tokens.empty()) {
        std::fprintf(stderr, "failed to tokenize resource or query\n");
        llama_model_free(model);
        return 1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = resource_tokens.size() + query_tokens.size() + 8;
    context_params.n_batch = resource_tokens.size() + query_tokens.size();
    context_params.n_seq_max = 3;
    // Unified sequence storage is required for metadata-only membership copy.
    // With separate streams llama.cpp schedules a physical buffer copy.
    context_params.kv_unified = true;
    context_params.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> full_tokens = resource_tokens;
    full_tokens.insert(full_tokens.end(), query_tokens.begin(), query_tokens.end());
    if (!decode_tokens(ctx, full_tokens, 0, 0, true)) {
        std::fprintf(stderr, "full-prefix decode failed\n");
        return 1;
    }
    const auto full_logits = last_logits(ctx, vocabulary_size);

    // Match the two-call resource/query schedule before changing sequence
    // membership. This separates ordinary split-prefill numerics from the
    // native selected-K/V mechanism under test and checks decode persistence.
    constexpr int32_t decode_steps = 4;
    const auto split = run_split_condition(
        ctx,
        memory,
        resource_tokens,
        query_tokens,
        vocabulary_size,
        false,
        decode_steps);
    const auto native = run_split_condition(
        ctx,
        memory,
        resource_tokens,
        query_tokens,
        vocabulary_size,
        true,
        decode_steps);
    if (split.logits.size() != decode_steps || native.logits.size() != decode_steps) {
        std::fprintf(stderr, "split-prefix or native persistent decode failed\n");
        return 1;
    }

    const auto full_top = argmax(full_logits);
    const auto split_top = split.generated.front();
    const auto native_top = native.generated.front();
    float persistent_error = 0.0f;
    for (int32_t step = 0; step < decode_steps; ++step) {
        persistent_error = std::max(
            persistent_error,
            maximum_error(split.logits[step], native.logits[step]));
    }
    const bool sequence_equal = split.generated == native.generated;
    const auto lifecycle = run_lifecycle_control(
        ctx,
        memory,
        resource_tokens,
        query_tokens,
        vocabulary_size);
    std::printf(
        "{\"resource_tokens\":%zu,\"query_tokens\":%zu,"
        "\"full_vs_native_max_logit_error\":%.9g,"
        "\"split_e0_vs_native_max_logit_error\":%.9g,"
        "\"persistent_decode_max_logit_error\":%.9g,"
        "\"decode_steps\":%d,\"decode_sequence_equal\":%s,"
        "\"absent_request_isolation_max_logit_error\":%.9g,"
        "\"absent_request_top_token_equal\":%s,"
        "\"warm_resource_reuse_max_logit_error\":%.9g,"
        "\"full_top_token\":%d,\"split_e0_top_token\":%d,"
        "\"native_top_token\":%d,\"full_top_token_equal\":%s,"
        "\"split_e0_top_token_equal\":%s,"
        "\"physical_kv_copy\":false}\n",
        resource_tokens.size(),
        query_tokens.size(),
        maximum_error(full_logits, native.logits.front()),
        maximum_error(split.logits.front(), native.logits.front()),
        persistent_error,
        decode_steps,
        sequence_equal ? "true" : "false",
        lifecycle.absent_isolation_error,
        lifecycle.absent_top_equal ? "true" : "false",
        lifecycle.reuse_error,
        full_top,
        split_top,
        native_top,
        full_top == native_top ? "true" : "false",
        split_top == native_top ? "true" : "false");

    llama_free(ctx);
    llama_model_free(model);
    return sequence_equal ? 0 : 3;
}
