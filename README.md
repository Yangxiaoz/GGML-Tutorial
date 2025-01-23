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
./simple-sgemm
```

### 添加其他demo

1. 复制ggml/examples文件夹下的任何样例或者你自己的其他工程至/demo目录下

2. 然后在/demo/CMakeLists.txt文件最后使用add_subdirectory()添加移植的样例。

3. 重新进行编译

### 二、文档目录：

| 名称  | 简介       | 地址|
|:---:|:----: |:---: |
| - | -|-|
| ... | -     |-      |


### 三、demo目录：

| 名称  | 简介       | 地址|
|:---:|:----: |:---: |
| 1. sgemm | 关于经典矩阵乘法的ggml实现|[Link](./demo/demo-sgemm/README.md)|
| ... | -     |-      |

