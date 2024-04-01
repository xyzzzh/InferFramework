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

// 计算图类，由多个计算节点和节点之间的数据流图组成
class RuntimeGraph {
public:
    // 初始化计算图
    RuntimeGraph(std::string param_path, std::string bin_path);

    // 构建计算图
    void build(const std::string &input_name, const std::string &output_name);

    // 设置结构文件
    void set_param_path(const std::string& param_path);

    // 设置权重文件
    void set_bin_path(const std::string& bin_path);

    // 返回结构文件
    const std::string &param_path() const;

    // 返回权重文件
    const std::string &bin_path() const;

    // 返回当前图状态
    const EGraphState &state() const;

    // 计算图的初始化
    bool init();

    const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

private:
    // 初始化计算图节点中的输入操作数
    static void init_graph_operators_input(
            const std::vector<pnnx::Operand *> &inputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator);

    // 初始化计算图节点中的输出操作数
    static void init_graph_operators_output(
            const std::vector<pnnx::Operand *> &outputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator);

    // 初始化计算图中的节点属性
    static void
    init_graph_attrs(const std::map<std::string, pnnx::Attribute> &attrs,
                   const std::shared_ptr<RuntimeOperator> &runtime_operator);

    // 初始化计算图中的节点参数
    static void
    init_graph_params(const std::map<std::string, pnnx::Parameter> &params,
                    const std::shared_ptr<RuntimeOperator> &runtime_operator);

    // 拓扑排序
    void ReverseTopo(const std::shared_ptr<RuntimeOperator> &root_op);
private:
    std::string m_input_name;  /// 计算图输入节点的名称
    std::string m_output_name; /// 计算图输出节点的名称
    std::string m_param_path;  /// 计算图的结构文件
    std::string m_bin_path;    /// 计算图的权重文件

    std::vector<std::shared_ptr<RuntimeOperator>> m_operators;
    std::map<std::string, std::shared_ptr<RuntimeOperator>> m_operators_maps;
    std::vector<std::shared_ptr<RuntimeOperator>> m_topo_operators;

    std::unique_ptr<pnnx::Graph> m_graph; /// pnnx的graph

    EGraphState m_state = EGraphState::EGS_NeedInit;
};


#endif //INFERFRAMEWORK_RUNTIMEGRAPH_HPP
