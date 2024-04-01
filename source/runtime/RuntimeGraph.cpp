//
// Created by xyzzzh on 2024/3/30.
//

#include "runtime/RuntimeGraph.hpp"

RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path) {
    this->m_param_path = std::move(param_path);
    this->m_bin_path = std::move(bin_path);
}

void RuntimeGraph::build(const std::string &input_name, const std::string &output_name) {
    if (this->m_state == EGraphState::EGS_Completed) {
        LOG(INFO) << "Model has been built already!";
        return;
    }

    if (this->m_state == EGraphState::EGS_NeedInit) {
        bool init_graph = this->init();
        LOG_IF(FATAL, !init_graph) << "Init graph failed!";
    }

    CHECK(this->m_state == EGraphState::EGS_NeedBuild) << "Graph status error, current state is " << int(this->m_state);
    LOG_IF(FATAL, this->m_operators.empty()) << "Graph operators is empty, may be no init";

    // 构建图关系
    for (const auto &curr_op: this->m_operators) {
        // 获取当前节点的所有后继节点的names，遍历根据next_op_name从operators_maps中插入所需要的节点
        const std::vector<std::string> &output_names = curr_op->m_output_names;
        for (const auto &_name: output_names) {
            const auto &output_op = this->m_operators_maps.find(_name);
            if (output_op != this->m_operators_maps.end()) {
                curr_op->m_output_operators.insert({_name, output_op->second});
            }
        }
    }

    // 初始化节点的输入和输出空间
    init_operator_input(this->m_operators);
    init_operator_output(this->m_graph->ops, this->m_operators);

    // 构建拓扑顺序
    this->m_topo_operators.clear();
    for (const auto &[_, op]: this->m_operators_maps) {
        // 根据输入节点构建拓扑排序
        if (op->m_type == "pnnx.Input" && !op->m_has_forward) {
            this->ReverseTopo(op);
        }
    }

    CHECK(this->m_topo_operators.size() == this->m_operators.size())
                    << "Build wrong topo queue";
    std::reverse(this->m_topo_operators.begin(), this->m_topo_operators.end());

    this->m_state = EGraphState::EGS_Completed;
    this->m_input_name = input_name;
    this->m_output_name = output_name;
    if (this->m_graph != nullptr) {
        this->m_graph.reset();
        this->m_graph = nullptr;
    }

}

void RuntimeGraph::set_param_path(const std::string &param_path) {
    this->m_param_path = param_path;
}

void RuntimeGraph::set_bin_path(const std::string &bin_path) {
    this->m_bin_path = bin_path;
}

const std::string &RuntimeGraph::param_path() const {
    return this->m_param_path;
}

// 返回权重文件
const std::string &RuntimeGraph::bin_path() const {
    return this->m_bin_path;
}

const EGraphState &RuntimeGraph::state() const {
    return this->m_state;
}

// 计算图的初始化
bool RuntimeGraph::init() {
    if (this->m_bin_path.empty() || this->m_param_path.empty()) {
        LOG(ERROR) << "The bin path or param path is empty";
        return false;
    }

    this->m_graph = std::make_unique<pnnx::Graph>();

    int load_res = this->m_graph->load(this->m_param_path, this->m_bin_path);
    if (load_res != 0) {
        LOG(ERROR) << "Can not find the param path or bin path: " << this->m_param_path
                   << " " << this->m_bin_path;
        return false;
    }
    std::vector<pnnx::Operator *> operators = this->m_graph->ops;
    if (operators.empty()) {
        LOG(ERROR) << "Can not read the layers' define";
        return false;
    }

    this->m_operators.clear();
    this->m_operators_maps.clear();

    for (const auto &op: operators) {
        if (op == nullptr) {
            LOG(ERROR) << "Meet the empty node";
            continue;
        } else {
            std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
            runtime_operator->m_name = op->name;
            runtime_operator->m_type = op->type;

            // 初始化算子中的input
            const auto &inputs = op->inputs;
            if (!inputs.empty()) {
                init_graph_operators_input(inputs, runtime_operator);
            }

            // 记录输出operand中的名称
            const auto &outputs = op->outputs;
            if (!outputs.empty()) {
                init_graph_operators_output(outputs, runtime_operator);
            }

            // 初始化算子中的attribute(权重)
            const auto &attrs = op->attrs;
            if (!attrs.empty()) {
                init_graph_attrs(attrs, runtime_operator);
            }

            // 初始化算子中的parameter
            const auto &params = op->params;
            if (!params.empty()) {
                init_graph_params(params, runtime_operator);
            }

            this->m_operators.push_back(runtime_operator);
            this->m_operators_maps.insert({runtime_operator->m_name, runtime_operator});
        }
    }
    this->m_state = EGraphState::EGS_NeedBuild;
    return true;
}

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

        // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=cp64 11=cp128 12=cp32
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
                LOG(FATAL) << "Unknown input operand type: " << input->type;
            }
        }
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
        // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
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
        switch (type) {
            case int(ERuntimeParameterType::ERPT_ParameterUnknown): {
                RuntimeParameter *runtime_parameter = new RuntimeParameter;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterBool): {
                RuntimeParameterBool *runtime_parameter = new RuntimeParameterBool;
                runtime_parameter->value = parameter.b;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterInt): {
                RuntimeParameterInt *runtime_parameter = new RuntimeParameterInt;
                runtime_parameter->value = parameter.i;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterFloat): {
                RuntimeParameterFloat *runtime_parameter = new RuntimeParameterFloat;
                runtime_parameter->value = parameter.f;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterString): {
                RuntimeParameterString *runtime_parameter = new RuntimeParameterString;
                runtime_parameter->value = parameter.s;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterIntArray): {
                RuntimeParameterIntArray *runtime_parameter =
                        new RuntimeParameterIntArray;
                runtime_parameter->value = parameter.ai;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }

            case int(ERuntimeParameterType::ERPT_ParameterFloatArray): {
                RuntimeParameterFloatArray *runtime_parameter =
                        new RuntimeParameterFloatArray;
                runtime_parameter->value = parameter.af;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }
            case int(ERuntimeParameterType::ERPT_ParameterStringArray): {
                RuntimeParameterStringArray *runtime_parameter =
                        new RuntimeParameterStringArray;
                runtime_parameter->value = parameter.as;
                runtime_operator->m_params.insert({name, runtime_parameter});
                break;
            }
            default: {
                LOG(FATAL) << "Unknown parameter type";
            }
        }
    }
}

void RuntimeGraph::ReverseTopo(const std::shared_ptr<RuntimeOperator> &root_op) {
    CHECK(root_op != nullptr) << "current operator is nullptr";
    root_op->m_has_forward = true;
    const auto &next_ops = root_op->m_output_operators;
    for (const auto &[_, op]: next_ops) {
        if (op != nullptr) {
            if (!op->m_has_forward) {
                this->ReverseTopo(op);
            }
        }
    }

    for (const auto &[_, op]: next_ops) {
        CHECK(op->m_has_forward);
    }
    this->m_topo_operators.push_back(root_op);
}

const std::vector<std::shared_ptr<RuntimeOperator>> &RuntimeGraph::get_topo_queues() const {
    return this->m_topo_operators;
}

