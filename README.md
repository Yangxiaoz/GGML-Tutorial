# GGML-Tutorial

GGML机器学习库学习指南与代码示例



## 一、How to Start

1. Download

```bash
git clone https://github.com/Yangxiaoz/GGML-Tutorial.git
cd GGML-Tutorial
```

2. CPU Build

```bash
#构建编译build文件
cmake -B build #（如果使用vscode中cmake插件，则此步骤可能自动被执行）
cmake --build build  -j 4
```

3. Using CUDA

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j 12 #cuda编译较慢，尽可能选择更多core进行编译
```

4. 运行demo

可执行文件生成于“/build/bin/”文件下
```bash
cd build/bin/
（仅CPU）
./simple-sgemm

(启用CUDA)
./simple-sgemm CUDA0/CPU  #可手动输入参数选择，未输入参数时默认使用第一个后端
```

### 添加其他demo

1. 复制ggml/examples文件夹下的任何样例或者你自己的其他工程至/demo目录下

2. 然后在/demo/CMakeLists.txt文件最后使用add_subdirectory()添加移植的样例。

3. 重新进行编译


### 二、demo目录：

| 名称  | 简介       | 地址|
|:---:|:----: |:---: |
| 1. sgemm | 关于经典矩阵乘法的ggml实现|[Link](./demo/demo-sgemm/README.md)|
| ... | -     |-      |

### 三、文档目录：

| 名称  | 简介       | 地址|
|:---:|:----: |:---: |
| Huggingface_ggml介绍 | 关于ggml基础概念于demo介绍|[Link](https://huggingface.co/blog/introduction-to-ggml)|
| GGML 核心概念 | 关于ggml核心概念解析|[Link](./doc/Core-Concepts.md)|
| Mnist手写数字识别demo | 关于ggml/examples/mnist手写数字识别demo源码流程思维导图|[Link](https://n02lxruxa4.feishu.cn/wiki/HPGjwT7FAiyZttkNCErcl7lXnKg?from=from_copylink)|
| ... | -     |-      |