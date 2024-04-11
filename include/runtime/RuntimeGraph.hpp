//
// Created by xyzzzh on 2024/3/30.
//

#ifndef INFERFRAMEWORK_RUNTIMEGRAPH_HPP
#define INFERFRAMEWORK_RUNTIMEGRAPH_HPP

#include "Common.hpp"
#include "Utils.hpp"
#include "runtime/RuntimeOperator.hpp"
#include "runtime/RuntimeOperand.hpp"
#include "runtime/RuntimeAttribute.hpp"
#include "runtime/RuntimeParameter.hpp"

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

#endif //INFERFRAMEWORK_RUNTIMEGRAPH_HPP
