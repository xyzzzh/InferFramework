# InferFramework

基于C++实现的深度学习CPU推理框架

## 项目结构

```
.
├── CMakeLists.txt
├── README.md
├── include
│       ├── data
│       ├── layer
│       ├── parser
│       └── runtime
├── log
├── main.cpp
├── source
│       ├── data
│       ├── layer
│       │       ├── abstract
│       │       └── details
│       ├── parser
│       └── runtime
└── test

```

## 依赖库

```
armadillo
glog
googletest
google-benchmark
```

## 参考资料
KuiperInfer https://github.com/zjhellofss/KuiperInfer/tree/main