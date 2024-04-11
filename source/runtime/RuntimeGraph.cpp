// Created by xyzzzh on 2024/3/30.
// 引入所需的头文件
#include "runtime/RuntimeGraph.hpp"
#include "layer/abstract/Layer.hpp"
#include "layer/abstract/LayerRegisterer.hpp"

// 构造函数，初始化参数路径和二进制文件路径
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path) {
    this->m_param_path = std::move(param_path);
    this->m_bin_path = std::move(bin_path);
}

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

// 设置参数路径
void RuntimeGraph::set_param_path(const std::string &param_path) {
    this->m_param_path = param_path;
}

// 设置二进制文件路径
void RuntimeGraph::set_bin_path(const std::string &bin_path) {
    this->m_bin_path = bin_path;
}

// 获取参数路径
const std::string &RuntimeGraph::param_path() const {
    return this->m_param_path;
}

// 获取二进制文件路径
const std::string &RuntimeGraph::bin_path() const {
    return this->m_bin_path;
}

// 获取计算图的状态
const EGraphState &RuntimeGraph::state() const {
    return this->m_state;
}

// 计算图的初始化函数
bool RuntimeGraph::init() {
    // 如果二进制文件路径或参数路径为空，则返回失败
    if (this->m_bin_path.empty() || this->m_param_path.empty()) {
        LOG(ERROR) << "The bin path or param path is empty";
        return false;
    }

    // 创建图对象
    this->m_graph = std::make_unique<pnnx::Graph>();

    // 加载参数和二进制文件
    int load_res = this->m_graph->load(this->m_param_path, this->m_bin_path);
    if (load_res != 0) {
        // 如果加载失败，记录错误并返回失败
        LOG(ERROR) << "Can not find the param path or bin path: " << this->m_param_path << " " << this->m_bin_path;
        return false;
    }

    // 获取图中的操作符
    std::vector<pnnx::Operator *> operators = this->m_graph->ops;
    if (operators.empty()) {
        // 如果操作符列表为空，记录错误并返回失败
        LOG(ERROR) << "Can not read the layers' define";
        return false;
    }

    // 清空当前的操作符列表和映射
    this->m_operators.clear();
    this->m_operators_maps.clear();

    // 遍历操作符，为每个操作符创建运行时表示，并初始化其属性
    for (const auto &op: operators) {
        if (op == nullptr) {
            LOG(ERROR) << "Meet the empty node";
            continue;
        } else {
            // 为每个操作符创建运行时表示
            std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
            runtime_operator->m_name = op->name;
            runtime_operator->m_type = op->type;

            // 初始化操作符的输入
            const auto &inputs = op->inputs;
            if (!inputs.empty()) {
                init_graph_operators_input(inputs, runtime_operator);
            }

            // 记录操作符的输出名称
            const auto &outputs = op->outputs;
            if (!outputs.empty()) {
                init_graph_operators_output(outputs, runtime_operator);
            }

            // 初始化操作符的属性
            const auto &attrs = op->attrs;
            if (!attrs.empty()) {
                init_graph_attrs(attrs, runtime_operator);
            }

            // 初始化操作符的参数
            const auto &params = op->params;
            if (!params.empty()) {
                init_graph_params(params, runtime_operator);
            }

            // 将操作符添加到列表和映射中
            this->m_operators.push_back(runtime_operator);
            this->m_operators_maps.insert({runtime_operator->m_name, runtime_operator});
        }
    }
    // 更新计算图的状态为需要构建
    this->m_state = EGraphState::EGS_NeedBuild;
    return true;
}

// 获取操作符列表
const std::vector<std::shared_ptr<RuntimeOperator>> &RuntimeGraph::operators() const {
    return this->m_operators;
}

// 初始化计算图节点中的输入操作数
void RuntimeGraph::init_graph_operators_input(
        const std::vector<pnnx::Operand *> &inputs,
        const std::shared_ptr<RuntimeOperator> &runtime_operator) {
    for (const auto input: inputs) {
        if (input == nullptr) {
            continue;
        }

        const pnnx::Operator *producer = input->producer;
        std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
        runtime_operand->m_name = producer->name;
        runtime_operand->m_shapes = std::vector<uint32_t>(input->shape.begin(), input->shape.end());

        // 根据输入的数据类型设置运行时数据类型
        switch (input->type) {
            case 1: {
                runtime_operand->m_type = ERuntimeDataType::ERDT_Float32;
                break;
            }
            case 0: {
                runtime_operand->m_type = ERuntimeDataType::ERDT_Unknown;
                break;
            }
            default: {
                // 如果遇到未知的数据类型，记录致命错误
                LOG(FATAL) << "Unknown input operand type: " << input->type;
            }
        }
        // 将运行时操作数添加到操作符的输入列表中
        runtime_operator->m_input_operands.insert({producer->name, runtime_operand});
        runtime_operator->m_input_operands_seq.push_back(runtime_operand);
    }
}

// 初始化计算图节点中的输出操作数
void RuntimeGraph::init_graph_operators_output(
        const std::vector<pnnx::Operand *> &outputs,
        const std::shared_ptr<RuntimeOperator> &runtime_operator) {
    for (const pnnx::Operand *output: outputs) {
        if (output == nullptr) {
            continue;
        }
        // 记录每个输出操作数的消费者名称
        const auto &consumers = output->consumers;
        for (const auto &c: consumers) {
            runtime_operator->m_output_names.push_back(c->name);
        }
    }
}

// 初始化计算图中的节点属性
void
RuntimeGraph::init_graph_attrs(const std::map<std::string, pnnx::Attribute> &attrs,
                               const std::shared_ptr<RuntimeOperator> &runtime_operator) {
    for (const auto &[name, attr]: attrs) {
        // 根据属性的数据类型创建运行时属性
        switch (attr.type) {
            case 1: {
                std::shared_ptr<RuntimeAttribute> runtime_attr = std::make_shared<RuntimeAttribute>();
                runtime_attr->m_type = ERuntimeDataType::ERDT_Float32;
                runtime_attr->m_weight_data = attr.data;
                runtime_attr->m_shapes = std::vector<uint32_t>(attr.shape.begin(), attr.shape.end());
                runtime_operator->m_attribute.insert({name, runtime_attr});
                break;
            }
            default: {
                // 如果遇到未知的属性类型，记录致命错误
                LOG(FATAL) << "Unknown attribute type: " << attr.type;
            }
        }
    }
}

// 初始化计算图中的节点参数
void
RuntimeGraph::init_graph_params(const std::map<std::string, pnnx::Parameter> &params,
                                const std::shared_ptr<RuntimeOperator> &runtime_operator) {
    for (const auto &pair: params) {
        const std::string &name = pair.first;
        const pnnx::Parameter &parameter = pair.second;
        const int type = parameter.type;
        // 根据参数的类型创建对应的运行时参数
        switch (type) {
            case int(ERuntimeParameterType::ERPT_ParameterUnknown): {
                std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterBool): {
                std::shared_ptr<RuntimeParameterBool> runtime_parameter = std::make_shared<RuntimeParameterBool>();
                runtime_parameter->value = parameter.b;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterInt): {
                std::shared_ptr<RuntimeParameterInt> runtime_parameter = std::make_shared<RuntimeParameterInt>();
                runtime_parameter->value = parameter.i;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterFloat): {
                std::shared_ptr<RuntimeParameterFloat> runtime_parameter = std::make_shared<RuntimeParameterFloat>();
                runtime_parameter->value = parameter.f;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterString): {
                std::shared_ptr<RuntimeParameterString> runtime_parameter = std::make_shared<RuntimeParameterString>();
                runtime_parameter->value = parameter.s;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterIntArray): {
                std::shared_ptr<RuntimeParameterIntArray> runtime_parameter = std::make_shared<RuntimeParameterIntArray>();
                runtime_parameter->value = parameter.ai;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterFloatArray): {
                std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter = std::make_shared<RuntimeParameterFloatArray>();
                runtime_parameter->value = parameter.af;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }
            case int(ERuntimeParameterType::ERPT_ParameterStringArray): {
                std::shared_ptr<RuntimeParameterStringArray> runtime_parameter = std::make_shared<RuntimeParameterStringArray>();
                runtime_parameter->value = parameter.as;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }
            default: {
                // 如果遇到未知的参数类型，记录致命错误
                LOG(FATAL) << "Unknown parameter type";
            }
        }
    }
}

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

// 获取拓扑排序的操作符列表
const std::vector<std::shared_ptr<RuntimeOperator>> &RuntimeGraph::get_topo_queues() const {
    return this->m_topo_operators;
}

// 根据操作符类型创建对应的层
std::shared_ptr<Layer> RuntimeGraph::create_layer(const std::shared_ptr<RuntimeOperator> &op) {
    LOG_IF(FATAL, !op) << "Operator is empty!";
    // 使用层注册器创建对应的层
    auto layer = LayerRegisterer::create_layer(op);
    LOG_IF(FATAL, !layer) << "Layer init failed " << op->m_type;
    return layer;
}

// 执行计算图的前向传播
std::vector<std::shared_ptr<Tensor>>
RuntimeGraph::forward(const std::vector<std::shared_ptr<Tensor>> &inputs, bool debug) {
    // 检查计算图是否已经完成构建
    if (this->m_state < EGraphState::EGS_Completed) {
        LOG(FATAL) << "Graph need be build!";
    }
    CHECK(this->m_state == EGraphState::EGS_Completed)
                    << "Graph status error, current state is " << int(this->m_state);

    CHECK(this->m_topo_operators.size() == this->m_operators.size())
                    << "Build wrong topo queue";

    // 重置所有操作符的前向标志
    for (const auto &op: this->m_topo_operators) {
        op->m_has_forward = false;
    }

    // 遍历拓扑排序的操作符列表，执行前向传播
    for (const auto &current_op: this->m_topo_operators) {
        if (current_op == nullptr) {
            LOG(INFO) << "current_op== nullptr";
        }
        // 如果当前操作符为输入节点，则将输入数据赋值给后继层
        if (current_op->m_type == "pnnx.Input") {
            current_op->m_has_forward = true;
            probe_next_layer(current_op, inputs);
        } else if (current_op->m_type == "pnnx.Output") {
            // 如果当前操作符为输出节点，则将前一层的输出作为计算图的输出
            current_op->m_has_forward = true;
            CHECK(current_op->m_input_operands_seq.size() == 1);
            current_op->m_output_operands = current_op->m_input_operands_seq.front();
        } else {
            // 对于普通层，执行层的前向传播
            EInferStatus status = current_op->m_layer->forward();
            CHECK(status == EInferStatus::EIS_InferSuccess)
                            << current_op->m_layer->layer_name()
                            << " layer forward failed, error code: " << int(status);
            // 标记当前操作符为已执行前向传播
            current_op->m_has_forward = true;
            // 将当前层的输出数据赋值给后继层
            probe_next_layer(current_op, current_op->m_output_operands->m_data);
        }
    }

    // 检查所有操作符是否都已执行前向传播
    for (const auto &op: this->m_topo_operators) {
        LOG_IF(FATAL, !op->m_has_forward)
                        << "The operator: " << op->m_name << " has not been forward yet!";
    }

    // 根据输出名称获取输出操作符，返回计算图的最终输出
    if (m_operators_maps.find(m_output_name) != m_operators_maps.end()) {
        const auto &output_op = m_operators_maps.at(m_output_name);
        CHECK(output_op->m_output_operands != nullptr)
                        << "Output from" << output_op->m_name << " is empty";
        const auto &output_operand = output_op->m_output_operands;
        return output_operand->m_data;
    } else {
        // 如果无法找到输出操作符，记录致命错误
        LOG(FATAL) << "Can not find the output operator " << m_output_name;
        return std::vector<std::shared_ptr<Tensor>>{};
    }
}

// 将当前层的输出数据赋值给后继层
void RuntimeGraph::probe_next_layer(const std::shared_ptr<RuntimeOperator> &current_op,
                                    const std::vector<std::shared_ptr<Tensor>> &layer_output_data) {
    // 获取当前节点的所有后继节点
    const auto &next_ops = current_op->m_output_operators;
    // 遍历所有后继节点
    for (const auto &[_, next_rt_operator]: next_ops) {
        // 获取后继节点的所有输入操作数
        const auto &next_input_operands = next_rt_operator->m_input_operands;
        // 检查后继节点的输入是否来自于当前操作符
        if (next_input_operands.find(current_op->m_name) !=
            next_input_operands.end()) {
            // 获取后继节点关于当前操作符输出的输入数据
            std::vector<std::shared_ptr<Tensor>> &next_input_datas =
                    next_input_operands.at(current_op->m_name)->m_data;
            CHECK(next_input_datas.size() == layer_output_data.size());
            // 将当前操作符的输出数据赋值给后继节点的输入数据
            for (int i = 0; i < next_input_datas.size(); ++i) {
                next_input_datas.at(i) = layer_output_data.at(i);
            }
        }
    }
}
