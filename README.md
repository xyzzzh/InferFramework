# InferFramework

本项目参考学习[KuiperInfer](https://github.com/zjhellofss/KuiperInfer/tree/main)，实现了一个基于C++的深度学习CPU推理框架。目前能够支持Resnet的推理。

## 项目结构

```
.
├── CMakeLists.txt
├── README.md
├── include                 // 头文件
│         ├── Common.hpp    // 公共头文件
│         ├── Utils.hpp     // 辅助函数
│         ├── data
│         │         └── Tensor.hpp  // 张量类
│         ├── fmath.hpp             // 第三方数学库
│         ├── layer
│         │         ├── abstract
│         │         │         ├── Layer.hpp             // Layer基类
│         │         │         ├── LayerRegisterer.hpp   // Layer注册类
│         │         │         ├── NonParamLayer.hpp     // 无参类（=Layer）
│         │         │         └── ParamLayer.hpp        // 有参类
│         │         └── deatil  // 算子具体实现
│         │             ├── AdaptiveAveragePoolingLayer.hpp // 自适应平均池化
│         │             ├── ConvLayer.hpp                   // 卷积
│         │             ├── ExpressionLayer.hpp             // 表达式
│         │             ├── FlattenLayer.hpp                // Flatten
│         │             ├── LinearLayer.hpp                 // 全连接
│         │             ├── MaxPoolingLayer.hpp             // 最大池化
│         │             ├── ReluLayer.hpp                   // ReLU
│         │             └── SoftmaxLayer.hpp                // SoftMax
│         ├── parser
│         │         └── ExpressionParser.hpp                // 表达式解析类
│         └── runtime
│             ├── RuntimeAttribute.hpp      // 计算图节点的属性信息
│             ├── RuntimeGraph.hpp          // 计算图类
│             ├── RuntimeOperand.hpp        // 计算图中的操作数
│             ├── RuntimeOperator.hpp       // 计算图中的计算节点
│             ├── RuntimeParameter.hpp      // 计算节点中的参数信息
│             ├── ir.h                      // pnnx
│             └── store_zip.hpp             // pnnx
├── log     // 日志
├── model_file  // 模型文件
│         ├── car.jpg
│         ├── resnet18_batch1.pnnx.bin
│         ├── resnet18_batch1.pnnx.param
│         ├── simple_ops.pnnx.bin
│         ├── simple_ops.pnnx.param
│         ├── simple_ops2.pnnx.bin
│         ├── simple_ops2.pnnx.param
│         ├── test_linear.pnnx.bin
│         └── test_linear.pnnx.param
├── source
│         ├── Utils.cpp
│         ├── data
│         │         └── Tensor.cpp
│         ├── layer
│         │         ├── abstract
│         │         │         ├── Layer.cpp
│         │         │         ├── LayerRegisterer.cpp
│         │         │         └── ParamLayer.cpp
│         │         └── detail
│         │             ├── AdaptiveAveragePoolingLayer.cpp
│         │             ├── ConvLayer.cpp
│         │             ├── ExpressionLayer.cpp
│         │             ├── FlattenLayer.cpp
│         │             ├── LinearLayer.cpp
│         │             ├── MaxPoolingLayer.cpp
│         │             ├── ReluLayer.cpp
│         │             └── SoftmaxLayer.cpp
│         ├── parser
│         │         └── ExpressionParser.cpp
│         └── runtime
│             ├── RuntimeGraph.cpp
│             ├── ir.cpp
│             └── store_zip.cpp
├── test    // 测试代码
│         ├── main.cpp
│         ├── test_conv.cpp
│         ├── test_expression.cpp
│         ├── test_graph.cpp
│         ├── test_layer_registry.cpp
│         ├── test_load_data.cpp
│         ├── test_maxpooling.cpp
│         ├── test_resnet.cpp
│         ├── test_tensor.cpp
│         └── test_topo.cpp
└── tmp
    └── data_loader
        ├── data1.csv
        ├── data2.csv
        ├── data3.csv
        ├── data4.csv
        ├── data5.csv
        ├── data6.csv
        └── data7.csv
```

## 依赖库

```
armadillo
glog
googletest
google-benchmark
opencv2
```

## 参考资料

KuiperInfer https://github.com/zjhellofss/KuiperInfer/tree/main