# Simple-sgemm

此示例执行了矩阵乘法与加法，用于演示 ggml 和后端处理的基本用法。代码已注释，以帮助理解每个部分的作用。

## SGEMM定义

SGEMM（Single-Precision General Matrix Multiply）是 BLAS（Basic Linear Algebra Subprograms）库中的一个常用函数，执行单精度矩阵乘法。常被当作矩阵优化测试样例。

在本例中样例为：
$$
C = A\times B + C
$$

$$
\begin{bmatrix}
2 & 8 \\
5 & 1 \\
4 & 2 \\
8 & 6 \\
\end{bmatrix}
\times
\begin{bmatrix}
10 & 9 & 5 \\
5 & 9 & 4 \\
\end{bmatrix}
+
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
\end{bmatrix}
\=
\begin{bmatrix}
61 & 56 & 51 & 111 \\
91 & 55 & 55 & 127\\
43 & 30 & 29 & 65 \\
\end{bmatrix}
$$


值得注意的是在 `ggml` 中，我们以转置形式传递矩阵 $B$，然后逐行相乘。结果 $C$ 也是转置的，如下所示：


$$
mul\_mat(A, B^T) = C^T
$$

$$
ggml\_mul\_mat(
\begin{bmatrix}
2 & 8 \\
5 & 1 \\
4 & 2 \\
8 & 6 \\
\end{bmatrix}
,
\begin{bmatrix}
10 & 5 \\
9 & 9 \\
5 & 4 \\
\end{bmatrix}
)
\=
\begin{bmatrix}
60 & 55 & 50 & 110 \\
90 & 54 & 54 & 126 \\
42 & 29 & 28 & 64 \\
\end{bmatrix}
$$

## 关于源码

在本源码中，你可以更改.cpp文件顶部的标志位来选择是否使用CPU并行加速

```c
#define Flag_CPU_Parallel       1
```

你可以通过修改宏来更改测试矩阵的维度：
```c
#define sgemm_M   4
#define sgemm_K   2
#define sgemm_N   3
```

一般情况下当你想要感受CPU、GPU的并行加速的魅力时，你需要将矩阵维度设置的很大，效果才明显。(如1024x2048)

但是当矩阵的维度较大时，不要忘记注释main函数底部的结果打印部分，否则会打印冲刷终端。