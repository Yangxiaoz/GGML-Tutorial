#include "custom-model.h"

#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <stdexcept>


// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

static ggml_backend_buffer_type_t select_weight_buft( const buft_list_t & buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        
        return cur_buft;
        
    }

    return nullptr;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev) {
    //TBD: chech if the dev is gpu??
    buft_list_t buft_list;
    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    return buft_list;
}


// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    // for (auto * dev : devices) {
    //     ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    //     if (buft) {
    //         buft_list.emplace_back(dev, buft);
    //         break;
    //     }
    // }

    // // add extra buffer types, only if no GPU device is present
    // // ref: https://github.com/ggml-org/llama.cpp/issues/12481#issuecomment-2743136094
    // auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    // auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    // auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
    //     ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    // if (ggml_backend_dev_get_extra_bufts_fn) {
    //     ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
    //     while (extra_bufts && *extra_bufts) {
    //         buft_list.emplace_back(cpu_dev, *extra_bufts);
    //         ++extra_bufts;
    //     }
    // }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }
    return buft_list;
}


// ggml_backend_dev_t * devices;

// int32_t  n_threads;         // number of threads to use for generation
// int32_t  n_threads_batch;   // number of threads to use for batch processing

// //inherit from llama_context
// ////////////////////////////
// ggml_backend_sched_eval_callback cb_eval;
// void * cb_eval_user_data;

// bool split_flag;
// bool debuf_flag;
// bool no_perf;     // whether to measure performance timings
custom_model_params custom_context_default_params() {
    custom_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.n_threads                   =*/ 8,
        /*.n_threads_batch             =*/ 8,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.run_mode                    =*/ run_Hybrid,
        /*.split_flag                   =*/true,
        /*.debuf_flag                   =*/true, 
        /*.no_perf                      =*/true,
    };

    return result;
}



struct custom_model::impl {
    impl() {}
    ~impl() {}

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;


    // contexts where the model tensors metadata is stored
    std::vector<ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_ptr> bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct tensor_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    tensor_dev dev_input  = {};
    tensor_dev dev_output = {};
    std::vector<tensor_dev> dev_weight;
};

bool custom_model:: load_weight(){
    int inp_byte = ggml_nbytes(inp);
    int inp_dim = inp_byte / sizeof(float); // 确保 len 是元素个数
    float data[inp_dim];

    std::fill(data, data + inp_dim, 1.0f); // 将输入设置为1
    ggml_backend_tensor_set(inp, data, 0, inp_byte);

    std::fill(data, data + inp_dim, 2.0f); // 将第一个乘法weight设置为2


    for(int i =0;i < weight_0_mm->ne[1]; i++){
        ggml_backend_tensor_set(weight_0_mm, data, i* weight_0_mm->nb[1], inp_byte);

    }
    std::fill(data, data + inp_dim, 2.0f); // 将第一个add 偏置设置为2
    for(int i =0;i < weight_1_add->ne[1]; i++){
        ggml_backend_tensor_set(weight_1_add, data, i* weight_1_add->nb[1], inp_byte);
    }

    std::fill(data, data + inp_dim, 0.5f); // 将第二个乘法weight设置为2
    //TBD: to breakpoint check
    for(int i =0;i < weight_2_mm->ne[1]; i++){
        ggml_backend_tensor_set(weight_2_mm, data, i* weight_2_mm->nb[1], inp_byte);

    }
    float temp = -inp_dim;
    std::fill(data, data + inp_dim, temp); // 将第一个乘法weight设置为2
    for(int i =0;i < weight_3_add->ne[1]; i++){
        ggml_backend_tensor_set(weight_3_add, data, i* weight_3_add->nb[1], inp_byte);
    }
    return true;
}


custom_model::custom_model(custom_model_params params): graph_backend_t(params.graph_mode),pimpl(std::make_unique<impl>()){
    
    cb_eval           =  params.cb_eval;
    cb_eval_user_data =  params.cb_eval_user_data;;
//device_init
    // create list of devices to use with this model
    //only  push acc device to *this.device:
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            devices.push_back(*dev);
        }
    } else {
        // use all available devices
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU:
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    devices.push_back(dev);
                    break;
            }
        }

    }

    //assign backend and buffer of the tensors according device_list
        //backend buffer set:
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu_dev = nullptr;
    pimpl->cpu_buft_list = make_cpu_buft_list(devices);
        //TBD: need to chech if the dev is  gpu??? //add gpu device
    for (auto * dev : devices) {
        gpu_dev = devices.at(0);//use the fist device as normally
        buft_list_t buft_list = make_gpu_buft_list(dev);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev, std::move(buft_list));
    }

    

        //assign weight:
    pimpl->dev_weight.resize(n_weight);
    switch (graph_backend_t)
    {
    case run_CPU:
        //assignment all weight to cpu
        for (int i = 0; i < n_weight; ++i) {
        pimpl->dev_weight[i] = {cpu_dev, &pimpl->cpu_buft_list};
        }
        break;
    case run_GPU:
        //assignment all weight to gpu
        GGML_ASSERT(!devices.empty() && "no gpu devices can be used!");
        for (int i = 0; i < n_weight; ++i) {
            GGML_ASSERT(gpu_dev != nullptr);
            pimpl->dev_weight[i] = {gpu_dev, &pimpl->gpu_buft_list.at(gpu_dev)};
        }
        break;
    case run_Hybrid:
        GGML_ASSERT(!devices.empty() && "no gpu devices can be used!");
        for (int i = 0; i < n_weight; ++i) {
            if(i < n_weight/2){
                pimpl->dev_weight[i] = {cpu_dev, &pimpl->cpu_buft_list};
            }else{
                GGML_ASSERT(gpu_dev != nullptr);
                pimpl->dev_weight[i] = {gpu_dev, &pimpl->gpu_buft_list.at(gpu_dev)};
            }
        }
        break;   
    default:
        break;
    }
        // assign the input 
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
        //assign output
    pimpl->dev_output = { cpu_dev, &pimpl->cpu_buft_list };


// create tensor 
    //set the ctx for difference buft
    int max_n_tensors = n_weight;
    max_n_tensors += 2;// duplicated inp & output tensor
    const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;
    // one ggml context per buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    //define a lambd func for map context
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context");
            }

            ctx_map[buft] = ctx;
            pimpl->ctxs.emplace_back(ctx);

            return ctx;
        }
        return it->second;
    };
    //define a lambd func for create tensor
    auto create_tensor = [&](tensor_type type,int weight_id,const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor * {
    int64_t shape[2];
    char buffer[64];  // a temp buffer use to concat strings
    const char * name = buffer;
    int i = 0;
    for (const auto& dim : ne) {
        shape[i++] = dim;
    }
    GGML_ASSERT(i == 2 && "ne of tensor must is 2 ");
    struct ggml_tensor * ret;
    buft_list_t * buft_list;
    ggml_backend_buffer_type_t buft;
    ggml_context * ctx;

    switch (type)
    {
    case tensor_type::input:
        buft_list = pimpl->dev_input.buft_list;
        //use the first as fualt
        GGML_ASSERT(!buft_list->empty());
        buft = buft_list->front().second;
        GGML_ASSERT(buft && "failed to find a compatible buffer type for tensor");
        ctx = ctx_for_buft(buft);
        ret = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[0], shape[1]);
        ggml_set_input(ret);
        snprintf(buffer, sizeof(buffer), "input");
        break;
    case tensor_type::weight:
        buft_list = pimpl->dev_weight[weight_id].buft_list;
        //use the first as fualt
        GGML_ASSERT(!buft_list->empty());
        buft = buft_list->front().second;
        GGML_ASSERT(buft && "failed to find a compatible buffer type for tensor");
        ctx = ctx_for_buft(buft);
        ret = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[0], shape[1]);
        snprintf(buffer, sizeof(buffer), "weight_%d", weight_id);
        break;
    case tensor_type::output:
        snprintf(buffer, sizeof(buffer), "output");
        break;   
    default:
        break;
    }
    GGML_ASSERT(ret != nullptr && "erro create tensor!");
    ggml_set_name(ret, name);
    return ret;
    };

    inp          = create_tensor(tensor_type::input, 0,{dim_inp,1},0);
    weight_0_mm  = create_tensor(tensor_type::weight, 0,{dim_inp,dim_inp},0);
    weight_1_add = create_tensor(tensor_type::weight,1,{dim_inp,1},0);
    weight_2_mm  = create_tensor(tensor_type::weight, 2,{dim_inp,dim_inp},0);
    weight_3_add = create_tensor(tensor_type::weight,3,{dim_inp,1},0);
    // res          = create_tensor(tensor_type::output, 0,{dim_inp,1},0);
    
    // create the backend buffers of ctx
    const size_t n_max_backend_buffer = ctx_map.size();
    pimpl->bufs.reserve(n_max_backend_buffer);

    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx              = it.second;

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf == nullptr) {
            throw std::runtime_error(("unable to allocate buffer"));
        }

        pimpl->bufs.emplace_back(buf);
        if (pimpl->bufs.empty()) {
            throw std::runtime_error("failed to allocate buffer");
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // print memory requirements per buffer type
    for (auto & buf : pimpl->bufs) {
        printf("%s: %12s model buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
    }
    // load tensor data
    if(!load_weight() ){
        printf("erro tensor weight set!!!\n");
        exit(EXIT_FAILURE);
    }

    /////////////
//TBD:from :llama_context

    //init teh backend
        // GPU backends
    for (auto * dev : devices) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            throw std::runtime_error("failed to initialize %s backend");
        }
        backends.emplace_back(backend);
    }

        // add CPU backend
    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend_cpu == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    backends.emplace_back(backend_cpu);
        // create a list of the set_n_threads functions in the backends
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
            }
        }
    }

    //：inital backend from llama_context, to be determine....
        // init backends
     {
        printf("%s: enumerating backends\n", __func__);
        backend_buft.clear();
        backend_ptrs.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                auto * dev = devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
        }

        printf("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());
        const size_t max_nodes = 65536;
        printf("%s: max_nodes = %zu\n", __func__, max_nodes);

        // buffer used to store the computation graph and the tensor meta data
        buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

        //TBD: need check this sched init func
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false,false));

    }


}

custom_model::~custom_model() = default;

int32_t custom_model::graph_max_nodes() const {
    return std::max<int32_t>(65536, 5*(n_weight + 2));
}

ggml_cgraph * custom_model::graph_init() {
    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    return ggml_new_graph_custom(ctx_compute.get(), graph_max_nodes(), false);
}

struct ggml_tensor * custom_model::build_graph(ggml_context * ctx, ggml_cgraph * gf){

    ggml_tensor * cur;
    // C = mut_mat(A,B) == B(A.T)
    cur = ggml_mul_mat(ctx, weight_0_mm, inp);
    ggml_set_name(cur, "mul_mat_0");

    cur = ggml_add(ctx,cur,weight_1_add);
    ggml_set_name(cur, "add_1");

    cur = ggml_mul_mat(ctx, weight_2_mm, cur);
    ggml_set_name(cur, "mul_mat_2");

    cur = ggml_add(ctx,cur,weight_3_add);
    ggml_set_name(cur, "add_3");
   
    ggml_build_forward_expand(gf, cur);


    return ggml_graph_node(gf, -1);
}

int custom_model::compute(){

    ggml_backend_sched_reset(sched.get());
    ggml_backend_sched_set_eval_callback(sched.get(), cb_eval, cb_eval_user_data);

    auto * gf = graph_init();
    auto res = build_graph(ctx_compute.get(), gf);

    // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

    ggml_backend_sched_alloc_graph(sched.get(), gf);
    
    int n_threads        = 10;
    ggml_threadpool_t tp = threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        set_threadpool_fn(backend_cpu, tp);
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    return status;
}