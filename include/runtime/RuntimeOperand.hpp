//
// Created by xyzzzh on 2024/3/30.
//

#ifndef INFERFRAMEWORK_RUNTIMEOPERAND_HPP
#define INFERFRAMEWORK_RUNTIMEOPERAND_HPP

#include "runtime/Common.hpp"

// 计算节点输入输出的操作数
struct RuntimeOperand {
    std::string _name;                                          // 操作数的名字
    std::vector<uint32_t> _shapes;                              // 操作数的形状
    std::vector<std::shared_ptr<Tensor>> _data;                 // 操作数的数据
    ERuntimeDataType _type = ERuntimeDataType::ERDT_Unknown;    // 操作数的类型
};


#endif //INFERFRAMEWORK_RUNTIMEOPERAND_HPP
