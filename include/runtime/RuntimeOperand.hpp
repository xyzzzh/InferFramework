//
// Created by xyzzzh on 2024/3/30.
//

#ifndef INFERFRAMEWORK_RUNTIMEOPERAND_HPP
#define INFERFRAMEWORK_RUNTIMEOPERAND_HPP

#include "Common.hpp"

// 计算节点输入输出的操作数
struct RuntimeOperand {
    std::string m_name;                                          // 操作数的名字
    std::vector<uint32_t> m_shapes;                              // 操作数的形状
    std::vector<std::shared_ptr<Tensor>> m_data;                 // 操作数的数据
    ERuntimeDataType m_type = ERuntimeDataType::ERDT_Unknown;    // 操作数的类型
};


#endif //INFERFRAMEWORK_RUNTIMEOPERAND_HPP
