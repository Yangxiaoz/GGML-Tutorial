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

#include "custom-model.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>



// multi-core parallel   flag
#define Flag_CPU_Parallel       1  // 1: leveraging multi-cores for computing when benckend is cpu


static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}


int main(int argc, char ** argv) {
    ggml_time_init();

    // const std::string backend = argc >= 2 ? argv[1] : "";

    ggml_backend_load_all();

    custom_model_params param = custom_context_default_params();
    param.graph_mode = run_GPU;
    custom_model model(param);



    ggml_cgraph * gf = model.graph_init();
    model.res =  model.build_graph(model.ctx_compute.get(), gf);
    ggml_graph_dump_dot(gf, NULL, "llama.dot");
    model.compute();
    
    std::vector<float> out_data(ggml_nelements(model.res));
    ggml_backend_tensor_get(model.res,out_data.data(), 0, ggml_nbytes(model.res));

    return 0;
}
