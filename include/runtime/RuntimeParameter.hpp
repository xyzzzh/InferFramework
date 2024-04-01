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
    RuntimeParameterInt() : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterInt) {

    }
    int value = 0;
};

struct RuntimeParameterFloat : public RuntimeParameter {
    RuntimeParameterFloat() : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterFloat) {

    }
    float value = 0.f;
};

struct RuntimeParameterString : public RuntimeParameter {
    RuntimeParameterString() : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterString) {

    }
    std::string value;
};

struct RuntimeParameterIntArray : public RuntimeParameter {
    RuntimeParameterIntArray() : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterIntArray) {

    }
    std::vector<int> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
    RuntimeParameterFloatArray() : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterFloatArray) {

    }
    std::vector<float> value;
};

struct RuntimeParameterStringArray : public RuntimeParameter {
    RuntimeParameterStringArray() : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterStringArray) {

    }
    std::vector<std::string> value;
};

struct RuntimeParameterBool : public RuntimeParameter {
    RuntimeParameterBool() : RuntimeParameter(ERuntimeParameterType::ERPT_ParameterBool) {

    }
    bool value = false;
};
#endif //INFERFRAMEWORK_RUNTIMEPARAMETER_HPP
