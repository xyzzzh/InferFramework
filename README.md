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
├── source  // 实现代码
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
```

## 项目分析

### 1. 张量

张量是一个多维数组，因为本项目面向深度学习的视觉任务，张量可以被设计为三维格式，依次为channels(通道数), rows(行数)，cols(
列数)。数据本身可包括双精度(double)、单精度(float)或整型(int)。本项目中选取float作为Tensor的数据类型。

本项目选取[Armadillo](https://arma.sourceforge.net/)作为底层数学库，在arma::
fcube基础上开发Tensor类。需要注意的是，在armadillo中默认的顺序就是列主序的，而Pytorch张量默认顺序是行主序的，所以在程序中需要进行一定适应和调整。

Tensor类的具体实现如下：

```c++
// 默认为float
class Tensor {
public:
// channels     张量的通道数     uint32_t
// rows         张量的行数       uint32_t
// cols         张量的列数       uint32_t

// 构造函数
explicit Tensor() = default;

explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

explicit Tensor(const std::vector<uint32_t> &shapes);

// 拷贝构造
Tensor(const Tensor &other);

// 移动构造
Tensor(Tensor &&other) noexcept;

// 移动赋值
Tensor &operator=(Tensor &&other) noexcept;

// 拷贝赋值
Tensor &operator=(const Tensor &other);

// 获取rows
uint32_t rows() const;

// 获取cols
uint32_t cols() const;

// 获取channels
uint32_t channels() const;

// 获取m_data中元素总数
uint32_t size() const;

// 设置m_data
void set_data(const arma::fcube &data);

// 获取m_data是否为空
bool empty();

// 返回张量中offset位置的元素
float index(uint32_t offset) const;

float &index(uint32_t offset);

// 获取形状{channels(), rows(), cols()}
std::vector<uint32_t> shapes() const;

// 获取m_raw_shape
const std::vector<uint32_t> raw_shapes() const;

// 获取m_data
arma::fcube &data();

const arma::fcube data() const;

// 返回张量第channel通道中的数据
arma::fmat &slice(uint32_t channel);

// 获取m_data中第channel通道的fmat数据
const arma::fmat &slice(uint32_t channel) const;

// 获取m_data[channel, row, col]
float at(uint32_t channel, uint32_t row, uint32_t col) const;

float &at(uint32_t channel, uint32_t row, uint32_t col);

// 在m_data四周填充padding_value
// pads中顺序依次为{上, 下, 左, 右}
void padding(const std::vector<uint32_t> &pads, float padding_value);

// 用value填充m_data
void fill(float value);

// 用给定的浮点数数组values填充m_data。
void fill(const std::vector<float> &values, bool row_major = true);

// 以常量1初始化张量
void ones();

// 以随机值初始化张量
void rand();

// 打印张量
void show();

// 打印张量shapes
void show_shapes();

// 张量的实际尺寸大小的reshape
void reshape(const std::vector<uint32_t> &shapes, bool row_major = false);

// 对m_data进行压平
void flatten(bool row_major = false);

// 对张量中的元素进行过滤
void transform(const std::function<float(float)> &filter);

// 返回一个深拷贝后的张量
std::shared_ptr<Tensor> clone();

// 返回数据的原始指针
float *raw_ptr();

// 返回第index个矩阵的起始地址
float *matrix_raw_ptr(uint32_t index);

// 返回Tensor内的所有数据
std::vector<float> values(bool row_major = true);

private:
// 根据给定的形状参数重新调整Tensor对象中的数据。
// shapes: 一个包含目标形状的数组，预期为[target_channels, target_rows, target_cols]。
void review(const std::vector<uint32_t> &shapes);

// 张量数据的实际尺寸大小
// {rows, cols, channels}
std::vector<uint32_t> m_raw_shapes;

// 张量数据
arma::fcube m_data;
};
```

### 2.计算图

#### 2.1 计算图定义

本项目采用的模型格式为pnnx（PyTorch Neural Network eXchange）。
pnnx的优势在于支持模板匹配、保留表达式的整体结构并支持大量图优化操作，如算子融合，常量折叠和消除，公共表达式消除等。

PNNX由图结构(Graph), 运算符(Operator)和操作数(Operand)这三种结构组成的，设计非常简洁。pnnx将模型导出为.param和.bin文件，分别对应网络结构和权重参数。
Graph负责管理图中的运算符（算子），Operand类用来表示计算图中的操作数，即与一个运算符有关的输入和输出张量。
Graph类的成员函数提供了方便的接口用来创建和访问操作符和操作数，以构建和遍历计算图。同时，它也是模型中运算符（算子）和操作数的集合。

##### 2.1.1 PNNX中的运算符结构(Operator)

PNNX中的运算符实现如下：

```c++
class Operator
{
public:
    std::vector<Operand*> inputs;
    std::vector<Operand*> outputs;

    std::string type;
    std::string name;

    std::vector<std::string> inputnames;
    std::map<std::string, Parameter> params;
    std::map<std::string, Attribute> attrs;
}; 
```

在PNNX中，Operator用来表示一个算子，它由以下几个部分组成：

1. inputs：类型为std::vector<operand>, 表示这个算子在计算过程中所需要的输入操作数operand；
2. outputs：类型为std::vector<operand>, 表示这个算子在计算过程中得到的输出操作数operand；
3. type和name类型均为std::string, 分别表示该运算符号的类型和名称；
4. params, 类型为std::map, 用于存放该运算符的所有参数（例如卷积运算符中的params中将存放stride, padding, kernel size等信息）；
5. attrs, 类型为std::map, 用于存放该运算符所需要的具体权重属性（例如卷积运算符中的attrs中就存放着卷积的权重和偏移量，通常是一个float32数组）。

##### 2.1.2 PNNX中的Attribute和Param结构

在PNNX中，权重数据结构(Attribute)和参数数据结构(Param)定义如下。它们通常与一个运算符相关联，例如Linear算子的in_features属性和weight权重。

```c++
class Parameter
{
    // 0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
    int type;
    ...
    ...
}
class Attribute
{
public:
    Attribute()
        : type(0)
    {
    }

    Attribute(const std::initializer_list<int>& shape, const std::vector<float>& t);

    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
    int type;
    std::vector<int> shape;
    ...
};
```

##### 2.1.3 PNNX中的操作数结构(Operand)

pnnx的操作数类实现如下：

```c++
class Operand
{
public:
    void remove_consumer(const Operator* c);
    Operator* producer;
    std::vector<Operator*> consumers;
    
    int type;
    std::vector<int> shape;

    std::string name;
    std::map<std::string, Parameter> params;
};
```

其中，需要注意的是操作数结构中的producer和customers, 分别表示产生这个操作数的算子和使用这个操作数的算子。
产生这个操作数的算子只能有一个，而使用这个操作数的算子可以有很多个。

##### 2.1.4 项目中的实现

本项目中对pnnx的Graph、operator和operand进行了封装，代码位于`./include/runtime`文件夹下。接下来将对其进行依次分析。

首先是RuntimeOperand操作数，包括名字、形状、基于Tensor的数据和操作数的数据类型

```c++
// 计算节点输入输出的操作数
struct RuntimeOperand {
    std::string m_name;                                          /// 操作数的名字
    std::vector<uint32_t> m_shapes;                              /// 操作数的形状
    std::vector<std::shared_ptr<Tensor>> m_data;                 /// 操作数的数据
    ERuntimeDataType m_type = ERuntimeDataType::ERDT_Unknown;    /// 操作数的类型
};
```

然后是RuntimeParameter计算节点参数信息，并根据不同的数据类型实现了不同的派生类。

```c++
struct RuntimeParameter { /// 计算节点中的参数信息
    virtual ~RuntimeParameter() = default;

    explicit RuntimeParameter(ERuntimeParameterType type = ERuntimeParameterType::ERPT_ParameterUnknown) : type(type) {}

    ERuntimeParameterType type = ERuntimeParameterType::ERPT_ParameterUnknown;
};

struct RuntimeParameterFloat : public RuntimeParameter {
    RuntimeParameterFloat()
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterFloat) {}

    explicit RuntimeParameterFloat(float param_value)
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterFloat),
              value(param_value) {}

    float value = 0.f;
};
```

接下来是RuntimeAttribute，是计算图节点的属性信息：

```c++
// 计算图节点的属性信息
struct RuntimeAttribute {
  // 节点中的权重参数，以char形式存储。
  std::vector<char> m_weight_data;

  // 节点的形状信息，存储为一个无符号整数向量。
  std::vector<uint32_t> m_shapes;

  // 节点中的数据类型，使用枚举ERuntimeDataType表示，默认为ERDT_Unknown。
  ERuntimeDataType m_type = ERuntimeDataType::ERDT_Unknown;

  // 从节点的权重数据中提取权重，并将其转换为指定的数据类型T的向量。
  // 如果need_clear_weight为true，则在提取权重数据后清除原始的权重数据。
  template<typename T>
  std::vector<T> get_weight_data(bool need_clear_weight = true) {
    // 确保权重数据不为空。
    CHECK(!this->m_weight_data.empty());

    // 确保数据类型不是未知的。
    CHECK(this->m_type != ERuntimeDataType::ERDT_Unknown);

    std::vector<T> weight; // 用于存储转换后的权重数据。

    // 根据数据类型处理权重数据。
    switch (this->m_type) {
      case ERuntimeDataType::ERDT_Float32: { // 如果数据类型为float32。
        // 验证模板类型T是否为float。
        const bool is_float = std::is_same_v<T, float>;
        CHECK(is_float);

        // 确保权重数据大小能整除float的大小。
        const uint32_t float_size = sizeof(float);
        const uint32_t weight_size = this->m_weight_data.size();
        CHECK(weight_size % float_size == 0);

        // 将权重数据从char数组转换为float数组。
        auto *start_ptr = reinterpret_cast<const float *>(this->m_weight_data.data());
        auto *end_ptr = start_ptr + weight_size / float_size;
        weight.insert(weight.end(), start_ptr, end_ptr);
        break;
    }
    default: { // 如果数据类型未知。
      LOG(FATAL) << "Unknown weight data type" << int(this->m_type);
     }
    }

    // 如果需要，则清除权重数据。
    if (need_clear_weight) {
      this->clear_weight();
      }
    return weight;
  }

  // 清除权重数据。
  void clear_weight() {
  if (!this->m_weight_data.empty()) {
  m_weight_data.clear();
  }
  }
};
```

最后是RuntimeOperator操作数

```C++
// 计算图中的计算节点
struct RuntimeOperator {

    bool m_has_forward = false;
    std::string m_name;      /// 计算节点的名称
    std::string m_type;      /// 计算节点的类型
    std::shared_ptr<Layer> m_layer;  /// 节点对应的计算Layer

    std::map<std::string, std::shared_ptr<RuntimeOperand>> m_input_operands;  /// 节点的输入操作数
    std::vector<std::shared_ptr<RuntimeOperand>> m_input_operands_seq;    /// 节点的输入操作数，顺序排列
    std::map<std::string, std::shared_ptr<RuntimeOperator>> m_output_operators;  /// 输出节点的名字和节点对应

    std::vector<std::string> m_output_names;  /// 节点的输出节点名称
    std::shared_ptr<RuntimeOperand> m_output_operands;  /// 节点的输出操作数

    std::map<std::string, std::shared_ptr<RuntimeParameter>> m_params;  /// 算子的参数信息
    std::map<std::string, std::shared_ptr<RuntimeAttribute>> m_attribute;  /// 算子的属性信息，内含权重信息

};
```

完成了RuntimeOperand操作数、RuntimeParameter计算节点参数信息、RuntimeAttribute计算图节点属性信息和RuntimeOperator操作数后，在此基础上可以实现计算图核心部分RuntimeGraph。

```c++
class RuntimeGraph {
public:
    // 使用指定的结构文件和权重文件初始化计算图。
    RuntimeGraph(std::string param_path, std::string bin_path);

    // 根据输入和输出节点的名称构建计算图。
    void build(const std::string &input_name, const std::string &output_name);

    // 设置计算图的结构文件路径。
    void set_param_path(const std::string &param_path);

    // 设置计算图的权重文件路径。
    void set_bin_path(const std::string &bin_path);

    // 获取计算图的结构文件路径。
    const std::string &param_path() const;

    // 获取计算图的权重文件路径。
    const std::string &bin_path() const;

    // 获取计算图的当前状态。
    const EGraphState &state() const;

    // 初始化计算图，加载结构和权重文件。
    bool init();

    // 获取计算图中的所有操作符节点。
    const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

    // 获取计算图中经过拓扑排序的操作符队列。
    const std::vector<std::shared_ptr<RuntimeOperator>> &get_topo_queues() const;

    // 根据计算图中的操作节点创建相应的Layer。
    static std::shared_ptr<Layer> create_layer(const std::shared_ptr<RuntimeOperator> &op);

    // 对计算图进行前向传播，返回输出Tensor。
    std::vector<std::shared_ptr<Tensor>> forward(const std::vector<std::shared_ptr<Tensor>> &inputs, bool debug);

private:
    // 初始化计算图节点中的输入操作数。
    static void init_graph_operators_input(
            const std::vector<pnnx::Operand *> &inputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator);

    // 初始化计算图节点中的输出操作数。
    static void init_graph_operators_output(
            const std::vector<pnnx::Operand *> &outputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator);

    // 初始化计算图中的节点属性。
    static void
    init_graph_attrs(const std::map<std::string, pnnx::Attribute> &attrs,
                     const std::shared_ptr<RuntimeOperator> &runtime_operator);

    // 初始化计算图中的节点参数。
    static void
    init_graph_params(const std::map<std::string, pnnx::Parameter> &params,
                      const std::shared_ptr<RuntimeOperator> &runtime_operator);

    // 对计算图进行拓扑排序。
    void ReverseTopo(const std::shared_ptr<RuntimeOperator> &root_op);

    // 探查并处理下一层的计算节点。
    static void probe_next_layer(const std::shared_ptr<RuntimeOperator> &current_op,
                                 const std::vector<std::shared_ptr<Tensor>> &layer_output_data);

    // 成员变量
    std::string m_input_name;  // 计算图输入节点的名称。
    std::string m_output_name; // 计算图输出节点的名称。
    std::string m_param_path;  // 计算图的结构文件路径。
    std::string m_bin_path;    // 计算图的权重文件路径。

    std::vector<std::shared_ptr<RuntimeOperator>> m_operators; // 计算图中的操作节点列表。
    std::map<std::string, std::shared_ptr<RuntimeOperator>> m_operators_maps; // 操作节点的映射表。
    std::vector<std::shared_ptr<RuntimeOperator>> m_topo_operators; // 经过拓扑排序的操作节点列表。

    std::unique_ptr<pnnx::Graph> m_graph; // 使用pnnx库的Graph对象来管理计算图。

    EGraphState m_state = EGraphState::EGS_NeedInit; // 计算图的当前状态。
};
```

在`bool RuntimeGraph::init()`中加载对应的pnnx模型文件，并初始化Graph中的RuntimeOperand、RuntimeParameter、RuntimeAttribute和RuntimeOperator。

#### 2.2 计算图的构建和执行顺序

`bool RuntimeGraph::init()`实现了构建计算图中每个计算节点，但仍缺少两个部分：

- 计算图中所有计算节点的执行顺序；
- 计算节点相关的输入输出张量初始化。

深度学习模型是一个有向无环图。对于**有向图结构中的节点**，可以认为是深度学习模型中的**计算节点（算子）**，而**有向图结构中的边**可以认为是算子之间**连接和前后依赖关系**。可以用拓扑排序来确定模型中的节点执行顺序。

```c++
// 使用深度优先搜索进行拓扑排序的逆向遍历
void RuntimeGraph::ReverseTopo(const std::shared_ptr<RuntimeOperator> &root_op) {
    CHECK(root_op != nullptr) << "current operator is nullptr";
    // 标记当前操作符为已遍历
    root_op->m_has_forward = true;
    // 遍历当前操作符的所有输出操作符
    const auto &next_ops = root_op->m_output_operators;
    for (const auto &[_, op]: next_ops) {
        // 如果输出操作符未被遍历，则递归遍历之
        if (op != nullptr) {
            if (!op->m_has_forward) {
                this->ReverseTopo(op);
            }
        }
    }

    // 确保所有后继节点都已被遍历
    for (const auto &[_, op]: next_ops) {
        CHECK(op->m_has_forward);
    }
    // 将当前操作符添加到拓扑排序列表中
    this->m_topo_operators.push_back(root_op);
}
```

该拓扑排序方法会得到从**输出节点到输入节点的数组**，完成调用后需要进行逆序处理。

除了确定执行顺序，还需要构建图关系。遍历所有操作符，获取后续操作符名字（string），从m_operators_maps中索引，并将对应的输出操作符插入到当前操作符的m_output_operators中。

在计算图的构建过程中，可以观察到一个特点，即前一个节点的输出operand是下一个节点的输入operand，所以输入空间可以直接复用前一个的输出空间，可通过智能指针`std::shared_ptr`直接指向同一块内存区域来实现，由此可以减少内存占用。另外，节点输入和输出操作数是在构建期完成的，这意味着在运行时无需进行频繁的内存申请及释放开销，增加了运行效率。

Graph的构建函数完整实现如下：

```c++

// 构建函数，负责根据输入和输出名称构建计算图
void RuntimeGraph::build(const std::string &input_name, const std::string &output_name) {
    // 如果计算图已经构建完成，则不再重复构建
    if (this->m_state == EGraphState::EGS_Completed) {
        LOG(INFO) << "Model has been built already!";
        return;
    }

    // 如果需要初始化，则先进行初始化操作
    if (this->m_state == EGraphState::EGS_NeedInit) {
        bool init_graph = this->init();
        // 初始化失败的情况下，记录致命错误
        LOG_IF(FATAL, !init_graph) << "Init graph failed!";
    }

    // 检查计算图的状态是否为需要构建
    CHECK(this->m_state == EGraphState::EGS_NeedBuild) << "Graph status error, current state is " << int(this->m_state);
    // 如果操作符列表为空，则记录致命错误
    LOG_IF(FATAL, this->m_operators.empty()) << "Graph operators is empty, may be no init";

    // 构建图关系
    // 遍历所有操作符，构建操作符之间的关系
    for (const auto &curr_op: this->m_operators) {
        // 获取当前操作符的所有输出操作符的名称
        const std::vector<std::string> &output_names = curr_op->m_output_names;
        for (const auto &_name: output_names) {
            // 在操作符映射中查找对应的输出操作符
            const auto &output_op = this->m_operators_maps.find(_name);
            // 如果找到，则在当前操作符中插入对应的输出操作符
            if (output_op != this->m_operators_maps.end()) {
                curr_op->m_output_operators.insert({_name, output_op->second});
            }
        }
    }

    // 除了输入和输出节点外，为每个操作符创建对应的层
    for (const auto &op: this->m_operators) {
        if (op->m_type != "pnnx.Input" && op->m_type != "pnnx.Output") {
            std::shared_ptr<Layer> layer = RuntimeGraph::create_layer(op);
            // 确保层创建成功
            CHECK(layer != nullptr) << "Layer " << op->m_name << " create failed!";
            if (layer) {
                // 为操作符设置对应的层，并初始化层的运行时操作符
                op->m_layer = layer;
                layer->set_runtime_operator(op);
            }
        }
    }

    // 初始化节点的输入和输出空间
    init_operator_input(this->m_operators);
    init_operator_output(this->m_graph->ops, this->m_operators);

    // 构建拓扑排序
    this->m_topo_operators.clear();
    for (const auto &[_, op]: this->m_operators_maps) {
        // 根据输入节点构建拓扑排序
        if (op->m_type == "pnnx.Input" && !op->m_has_forward) {
            this->ReverseTopo(op);
        }
    }

    // 确保拓扑排序的大小与操作符列表的大小相同
    CHECK(this->m_topo_operators.size() == this->m_operators.size()) << "Build wrong topo queue";
    // 将拓扑排序反转以满足执行顺序
    std::reverse(this->m_topo_operators.begin(), this->m_topo_operators.end());

    // 更新计算图状态为已完成，记录输入和输出名称
    this->m_state = EGraphState::EGS_Completed;
    this->m_input_name = input_name;
    this->m_output_name = output_name;
    // 重置图对象
    if (this->m_graph != nullptr) {
        this->m_graph.reset();
        this->m_graph = nullptr;
    }
}
```

### 3. 算子层

#### 3.1 算子定义和注册

##### 3.1.1 算子定义

计算节点(`RuntimeOperator`)记录了与该节点相关的类型、名称，以及输入输出数等信息。其中最重要的是`Layer`变量，它表示与计算节点关联的算子，也就是进行具体计算的实施者。

首先，设计一个`Layer`基类，如果要实现项目中其他的算子，都需要继承于该类作为派生类并重写其中的计算函数(`forward`)。

`Layer`基类定义如下：

```c++

class Layer {
public:
    explicit Layer(std::string layer_name) : m_layer_name(std::move(layer_name)) {}

    virtual ~Layer() = default;

    // Layer的执行函数
    virtual EInferStatus forward(
            const std::vector<std::shared_ptr<Tensor>> &inputs,
            std::vector<std::shared_ptr<Tensor>> &outputs
    );

public:
    // Layer的执行函数
    virtual EInferStatus forward();

    // 返回层的权重
    virtual const std::vector<std::shared_ptr<Tensor>> &weights() const;

    // 返回层的偏移量
    virtual const std::vector<std::shared_ptr<Tensor>> &bias() const;

    // 设置Layer的权重
    virtual void set_weights(const std::vector<std::shared_ptr<Tensor>> &weights);

    virtual void set_weights(const std::vector<float> &weights);

    // 设置Layer的偏移量
    virtual void set_bias(const std::vector<std::shared_ptr<Tensor>> &bias);

    virtual void set_bias(const std::vector<float> &bias);

    // 返回层的名称
    virtual const std::string layer_name() const { return this->m_layer_name; }

    // 设置层的执行算子
    void set_runtime_operator(
            const std::shared_ptr<RuntimeOperator> &runtime_operator
    ) { this->m_runtime_operator = runtime_operator; }

private:
    std::weak_ptr<RuntimeOperator> m_runtime_operator;
    std::string m_layer_name;
};
```

不难看出，`RuntimeOperator`与该节点对应的 `Layer` 相关联，而 `Layer` 也关联了它所属的 `RuntimeOperator`，因此它们之间是双向关联的关系。

另外，在Layer中，有两个forward方法，一个是无参的外部调用接口，另一个是实际的内部实现。不带参数的 `forward` 方法是所有算子的父类方法，**它的作用是准备输入和输出数据，并使用这些数据调用每个派生类算子中各自实现的计算过程（带参数的forward）。**

##### 3.1.2 算子注册

本项目中算子注册机制使用了单例模式和工厂模式。首先，在全局范围内创建一个唯一的注册表`registry`，它是一个`map`类型的对象。**这个注册表的键是算子的类型，而值是算子的初始化过程。**

开发者完成一个算子的开发后，需要通过特定的注册机制将算子写入全局注册表中。这可以通过在注册表中添加键值对来实现。算子的类型作为键，算子的初始化过程作为值。这样，当需要使用某个算子时，可以根据算子的类型从全局注册表中方便地获取对应的算子。

在实现上单例模式确保了只有一个全局注册表实例，并且可以在代码的任何地方访问该注册表。工厂模式则负责根据算子的类型返回相应的算子实例。这种注册机制的设计使得推理框架能够感知到开发者已经实现的算子，并且能够方便地调用和使用这些算子。

算子具体注册过程如下：

1. **创建Creator函数**：首先定义一个符合`Creator`签名的函数，该函数能够根据给定的`RuntimeOperator`创建一个新的Layer实例，并返回创建状态。
2. **注册Creator函数**：通过调用`LayerRegisterer::register_creator`方法，将该Creator函数与一个Layer类型关联起来。或者，创建一个`LayerRegistererWrapper`实例，自动完成注册过程。
3. **使用注册表**：当需要根据`RuntimeOperator`创建Layer时，`LayerRegisterer::create_layer`方法会查找注册表，找到对应的Creator函数，并使用它来创建Layer实例。

总的来说，这个注册机制允许程序在运行时动态地创建不同类型的Layer对象，无需在代码中硬编码对象创建逻辑。通过使用函数指针和注册表，可以灵活地添加或修改Layer类型，从而提高代码的可扩展性和维护性。

算子注册代码定义如下：

```c++

class LayerRegisterer {
public:
    using Creator = std::function<EParseParameterAttrStatus(const std::shared_ptr<RuntimeOperator> &,
                                                            std::shared_ptr<Layer> &)>;
    using CreateRegistry = std::map<std::string, Creator>;


    // 向注册表注册算子
    static void register_creator(const std::string &layer_type, const Creator &creator);

    // 通过算子参数op来初始化Layer
    static std::shared_ptr<Layer> create_layer(const std::shared_ptr<RuntimeOperator> &op);

    // 返回算子的注册表
    static CreateRegistry &get_registry();

    // 返回所有已被注册算子的类型
    static std::vector<std::string> layer_types();

    static bool compare_CreateRegistry(const CreateRegistry& c1, const CreateRegistry& c2);
};

class LayerRegistererWrapper {
public:
    LayerRegistererWrapper(const std::string &layer_type,
                           const LayerRegisterer::Creator &creator) {
        LayerRegisterer::register_creator(layer_type, creator);
    }
};
```

#### 3.2 算子实现

##### 3.2.1 有参层

TODO

##### 3.2.2 无参层

TODO

##### 3.2.3 Conv卷积层

TODO

#### 3.3 表达式解析及表达式层

##### 3.3.1 表达式的定义及解析

`PNNX`中的表达式就是一个二元的计算过程，类似如下：

```Plaintext
output_mid = input1 + input2;
output = output_mid * input3;
```

在`PNNX`的表达式层（Expression Layer）中，提供了一种计算表达式，该表达式能够在一定程度上折叠计算过程并消除中间变量。例如，在残差结构中的add操作在`PNNX`中就是一个表达式层。

下面是`PNNX`中对上述过程的计算表达式表示，其中的`@0`和`@1`代表之前提到的计算数`RuntimeOperand`，用于表示计算表达式中的输入节点。

```Plaintext
mul(@2, add(@0, @1));
```

尽管这个抽象表达式看起来比较简单，但实际上可能存在更为复杂的情况，例如以下的例子。因此需要一个强大而可靠的表达式解析和语法树构建功能。

```
add(add(mul(@0, @1), mul(@2, add(add(add(@0, @2), @3), @4))), @5);
```

##### 3.3.2 表达式层



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