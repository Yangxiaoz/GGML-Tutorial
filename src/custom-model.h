#pragma once

// #include "llama.h"
// #include "llama-arch.h"
// #include "llama-graph.h"
// #include "llama-hparams.h"
// #include "llama-memory.h"
// #include "llama-vocab.h"
// #include <memory>

#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

enum tensor_type{
    input,
    weight,
    output,
};

enum run_mode {
    run_CPU,
    run_GPU,
    run_Hybrid//CPU+GPU
};
// lists of buffer types used for each layer

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

struct custom_model_params custom_context_default_params(void);
// static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices);

struct custom_model_params {
    // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
    ggml_backend_dev_t * devices;

    int32_t  n_threads;         // number of threads to use for generation
    int32_t  n_threads_batch;   // number of threads to use for batch processing


    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    run_mode graph_mode;
    // llama_progress_callback progress_callback;
    // void * progress_callback_user_data;
    bool split_flag;
    bool debuf_flag;
    bool no_perf;     // whether to measure performance timings

};

struct custom_model {

    explicit custom_model(custom_model_params params);
    ~custom_model();
    std::string name = "n/a";
//menbers:
    //tensor:
    struct ggml_tensor * inp   = nullptr;
    struct ggml_tensor * weight_0_mm  = nullptr;
    struct ggml_tensor * weight_1_add   = nullptr;
    struct ggml_tensor * weight_2_mm   = nullptr;
    struct ggml_tensor * weight_3_add = nullptr;
    struct ggml_tensor * res          = nullptr;
    int n_weight = 4;  //equal all num of tensors (execpt inp & output
    int dim_inp = 400;//dim of inp tensor 
    // list of devices used in this model
    std::vector<ggml_backend_dev_t> devices;
    ggml_context_ptr ctx_compute;
    
    run_mode graph_backend_t;
    // for quantize-stats only
    // std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;
    // int64_t t_load_us  = 0;
    // int64_t t_start_us = 0;

//func:
    bool load_weight();

    int32_t graph_max_nodes() const;
    ggml_cgraph * graph_init();
    struct ggml_tensor * build_graph(ggml_context * ctx,ggml_cgraph * gf);
    int compute();
    void res_extract();
    //compute(const simple_model & model, ggml_gallocr_t allocr)
private:
    struct impl;
    std::unique_ptr<impl> pimpl;

    ggml_backend_sched_ptr sched;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    
    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    
};


    // llm_graph_result_ptr build_graph(
    //         const llm_graph_params & params,
    //                    ggml_cgraph * gf,
    //                 llm_graph_type   type) const;

    // void load_stats  (llama_model_loader & ml);
    // void load_arch   (llama_model_loader & ml);
    // void load_hparams(llama_model_loader & ml);
    // void load_vocab  (llama_model_loader & ml);
    // bool load_tensors(llama_model_loader & ml); // returns false if cancelled by progress_callback

    // std::string arch_name() const;
    // std::string type_name() const;

    // std::string desc() const;

    // size_t size() const;
    // size_t n_tensors() const;
    // size_t n_devices() const;

    // // total number of parameters in the model
    // uint64_t n_elements() const;

    // void print_info() const;

    // ggml_backend_dev_t dev_layer(int il) const;
    // ggml_backend_dev_t dev_output() const;

    // ggml_backend_buffer_type_t select_buft(int il) const;

    // const struct ggml_tensor * get_tensor(const char * name) const;

    // // TODO: move this to new llm_arch_model_i interface
    // llama_memory_i * create_memory() const; // TODO: params

    // // TODO: move this to new llm_arch_model_i interface
