//
// Created by xyzzzh on 2024/3/30.
//

#ifndef INFERFRAMEWORK_RUNTIMEPARAMETER_HPP
#define INFERFRAMEWORK_RUNTIMEPARAMETER_HPP

#include "Common.hpp"

// 计算节点中的参数信息
struct RuntimeParameter { /// 计算节点中的参数信息
    virtual ~RuntimeParameter() = default;

    explicit RuntimeParameter(ERuntimeParameterType type = ERuntimeParameterType::ERPT_ParameterUnknown) : m_type(type) {

    }

    ERuntimeParameterType m_type = ERuntimeParameterType::ERPT_ParameterUnknown;
    std::variant<bool, int, float, std::string, std::vector<int>, std::vector<float>, std::vector<std::string>> m_value;

    template<typename T>
    inline void set_value(const T &val) { m_value = val; }

    template<typename T>
    inline T get_value() const { return std::get<T>(m_value); }
};


#endif //INFERFRAMEWORK_RUNTIMEPARAMETER_HPP
