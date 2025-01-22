#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>

//Function declaration
struct simple_model; 
simple_model load_model(float *A, float *B, float *C, int M, int K, int N, const std::string &backend);
struct ggml_cgraph * build_graph(const simple_model& model);
struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr);

// multi-core parallel   flag
#define Flag_CPU_Parallel       1  // 1: leveraging multi-cores for computing when benckend is cpu

// there are three matrix: A 、B and C  in sgemm_demo: " A X B.T + C "
#define num_Matrix 3
#define sgemm_M   4
#define sgemm_K   2
#define sgemm_N   3
float matrix_A[sgemm_M * sgemm_K] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };
float matrix_B[sgemm_N * sgemm_K] = {
        10, 5,
        9, 9,
        5, 4
    };// note that The B matrix will be automatically transposed during calculation
float matrix_C[sgemm_N * sgemm_M] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };// result matrix_C is transpose of (A X B.T)


static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}


//defined of sgemm model
struct simple_model {
    struct ggml_tensor * A;
    struct ggml_tensor * B;
    struct ggml_tensor * C;
    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t backend = NULL;
    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer;
    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
    // the compute graph (which is piont to a static variable of func, so no need to free )
    struct ggml_cgraph * gf;

    simple_model(const std::string & backend_name){
        int num_tensors = num_Matrix;
        struct ggml_init_params params {
                /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
        };
        // create context
        ctx = ggml_init(params);
        // create tensors

        A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sgemm_K, sgemm_M);
        B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sgemm_K, sgemm_N);
        C = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sgemm_M, sgemm_N);

        // init backend device:
        if (!backend_name.empty()) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
            if (dev == nullptr) {
                fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev_i = ggml_backend_dev_get(i);
                    fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(dev_i), ggml_backend_dev_description(dev_i));
                }
                exit(1);
            }
            backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);
    }
    else{//No device specified, select the first one
        ggml_backend_dev_t dev = ggml_backend_dev_get(0);
        backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend);
    }
    fprintf(stdout, "Using the %s as backend to computing...\n",ggml_backend_name(backend));

    // create a backend buffer (backend memory) and alloc the tensors from the context
    buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
    }

    ~simple_model(){
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
    }
};

// initialize the tensors of the model in this case 
simple_model load_model(float * A, float * B, float * C,int M, int K, int N,const std::string & backend) {
    //init the model
    simple_model model(backend);

    ggml_log_set(ggml_log_callback_default, nullptr);

    GGML_ASSERT(M == sgemm_M);
    GGML_ASSERT(K == sgemm_K);
    GGML_ASSERT(N == sgemm_N);
    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.A, A, 0, ggml_nbytes(model.A));
    ggml_backend_tensor_set(model.B, B, 0, ggml_nbytes(model.B));
    ggml_backend_tensor_set(model.C, C, 0, ggml_nbytes(model.C));
    return model;
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model) {

    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    // result： C = C + A*B^T
    struct ggml_tensor * result = ggml_add_inplace(ctx0,model.C,ggml_mul_mat(ctx0, model.A, model.B));
    
    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr) {

    // allocate tensors of graph
    ggml_gallocr_alloc_graph(allocr, model.gf);

    //if enable the parallel when using cpu backend
    if ((ggml_backend_is_cpu(model.backend)) && Flag_CPU_Parallel) {
        const int ncores_logical = std::thread::hardware_concurrency();
        const int nthreads = std::min(ncores_logical, (ncores_logical + 4) / 2);
        fprintf(stdout, "Enable the cpu parallel mode, and using %d threads  \n",nthreads);
        ggml_backend_cpu_set_n_threads(model.backend, nthreads);
    }
    const int64_t t_start_us = ggml_time_us();
    //Start calculating based on the graph
    ggml_backend_graph_compute(model.backend, model.gf);

    const int64_t t_load_us = ggml_time_us() - t_start_us;
    fprintf(stdout, "%s: compute model in %.2lf ms\n", __func__, t_load_us / 1000.0);
    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(model.gf, -1);
}





int main(int argc, char ** argv) {
    ggml_time_init();

    const std::string backend = argc >= 2 ? argv[1] : "";

    simple_model model = load_model(matrix_A, matrix_B, matrix_C, sgemm_M, sgemm_K, sgemm_N, backend);

    // calculate the temporaly memory required to compute
    ggml_gallocr_t allocr = NULL;
    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // Build graphs based on tensor relationships
        model.gf = build_graph(model);
        //reserve memory of backend for graph
        ggml_gallocr_reserve(allocr, model.gf);

        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }

    // perform computation
    struct ggml_tensor * result = compute(model, allocr);

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    
    // expected result:
    // [ 61.00 56.00 51.00 111.00
    //  91.00 55.00 55.00 127.00
    //  43.00 30.00 29.00 65.00 ]
    
    // print the result when verifying
    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");

    // release backend memory used for computation
    ggml_gallocr_free(allocr);
    return 0;
}
