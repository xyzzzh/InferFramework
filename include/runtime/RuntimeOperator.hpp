//
// Created by xyzzzh on 2024/3/30.
//

#ifndef INFERFRAMEWORK_RUNTIMEOPERATOR_HPP
#define INFERFRAMEWORK_RUNTIMEOPERATOR_HPP

#include "Common.hpp"
#include "runtime/RuntimeOperand.hpp"
#include "runtime/RuntimeParameter.hpp"
#include "runtime/RuntimeAttribute.hpp"

struct Layer;
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

    std::map<std::string, RuntimeParameter> m_params;  /// 算子的参数信息
    std::map<std::string, std::shared_ptr<RuntimeAttribute>> m_attribute;  /// 算子的属性信息，内含权重信息

};

#endif //INFERFRAMEWORK_RUNTIMEOPERATOR_HPP
