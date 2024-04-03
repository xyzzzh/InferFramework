//
// Created by xyzzzh on 2024/3/30.
//

#ifndef INFERFRAMEWORK_RUNTIMEPARAMETER_HPP
#define INFERFRAMEWORK_RUNTIMEPARAMETER_HPP

#include "Common.hpp"

struct RuntimeParameter { /// 计算节点中的参数信息
    virtual ~RuntimeParameter() = default;

    explicit RuntimeParameter(ERuntimeParameterType type = ERuntimeParameterType::ERPT_ParameterUnknown) : type(type) {

    }

    ERuntimeParameterType type = ERuntimeParameterType::ERPT_ParameterUnknown;
};

struct RuntimeParameterInt : public RuntimeParameter {
    RuntimeParameterInt()
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterInt) {}

    explicit RuntimeParameterInt(int param_value)
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterInt),
              value(param_value) {}

    int value = 0;
};

struct RuntimeParameterFloat : public RuntimeParameter {
    RuntimeParameterFloat()
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterFloat) {}

    explicit RuntimeParameterFloat(float param_value)
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterFloat),
              value(param_value) {}

    float value = 0.f;
};

struct RuntimeParameterString : public RuntimeParameter {
    RuntimeParameterString()
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterString) {}

    explicit RuntimeParameterString(std::string param_value)
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterString),
              value(std::move(param_value)) {}

    std::string value;
};

struct RuntimeParameterIntArray : public RuntimeParameter {
    RuntimeParameterIntArray()
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterIntArray) {}

    explicit RuntimeParameterIntArray(std::vector<int> param_value)
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterIntArray),
              value(std::move(param_value)) {}

    std::vector<int> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
    RuntimeParameterFloatArray()
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterFloatArray) {}

    explicit RuntimeParameterFloatArray(std::vector<float> param_value)
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterFloatArray),
              value(std::move(param_value)) {}

    std::vector<float> value;
};

struct RuntimeParameterStringArray : public RuntimeParameter {
    RuntimeParameterStringArray()
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterStringArray) {}

    explicit RuntimeParameterStringArray(std::vector<std::string> param_value)
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterStringArray),
              value(std::move(param_value)) {}

    std::vector<std::string> value;
};

struct RuntimeParameterBool : public RuntimeParameter {
    RuntimeParameterBool()
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterBool) {}

    explicit RuntimeParameterBool(bool param_value)
            : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterBool),
              value(param_value) {}

    bool value = false;
};

#endif //INFERFRAMEWORK_RUNTIMEPARAMETER_HPP
