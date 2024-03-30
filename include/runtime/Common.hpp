//
// Created by xyzzzh on 2024/3/29.
//

#ifndef INFERFRAMEWORK_COMMON_HPP
#define INFERFRAMEWORK_COMMON_HPP

#include <vector>
#include <string>
#include <memory>
#include <glog/logging.h>
#include <map>
#include <queue>
#include "data/Tensor.hpp"
#include "runtime/ir.h"
#include "runtime/store_zip.hpp"

// 计算节点属性中的权重类型
enum class ERuntimeDataType {
    ERDT_Unknown = 0,
    ERDT_Float32 = 1,
    ERDT_Float64 = 2,
    ERDT_Float16 = 3,
    ERDT_Int8 = 4
};

enum class ERuntimeParameterType {
    ERPT_ParameterUnknown = 0,
    ERPT_ParameterBool = 1,
    ERPT_ParameterInt = 2,

    ERPT_ParameterFloat = 3,
    ERPT_ParameterString = 4,
    ERPT_ParameterIntArray = 5,
    ERPT_ParameterFloatArray = 6,
    ERPT_ParameterStringArray = 7,
};

// 推理状态
enum class EInferStatus {
    EIS_InferUnknown = -1,
    EIS_InferSuccess = 0,

    EIS_InferFailedInputEmpty = 1,
    EIS_InferFailedWeightParameterError = 2,
    EIS_InferFailedBiasParameterError = 3,
    EIS_InferFailedStrideParameterError = 4,
    EIS_InferFailedDimensionParameterError = 5,
    EIS_InferFailedInputOutSizeMatchError = 6,

    EIS_InferFailedOutputSizeError = 7,
    EIS_InferFailedShapeParameterError = 9,
    EIS_InferFailedChannelParameterError = 10,
    EIS_InferFailedOutputEmpty = 11,

};

enum class EParseParameterAttrStatus {
    EPPAS_ParameterMissingUnknown = -1,
    EPPAS_ParameterMissingStride = 1,
    EPPAS_ParameterMissingPadding = 2,
    EPPAS_ParameterMissingKernel = 3,
    EPPAS_ParameterMissingUseBias = 4,
    EPPAS_ParameterMissingInChannel = 5,
    EPPAS_ParameterMissingOutChannel = 6,

    EPPAS_ParameterMissingEps = 7,
    EPPAS_ParameterMissingNumFeatures = 8,
    EPPAS_ParameterMissingDim = 9,
    EPPAS_ParameterMissingExpr = 10,
    EPPAS_ParameterMissingOutHW = 11,
    EPPAS_ParameterMissingShape = 12,
    EPPAS_ParameterMissingGroups = 13,
    EPPAS_ParameterMissingScale = 14,
    EPPAS_ParameterMissingResizeMode = 15,
    EPPAS_ParameterMissingDilation = 16,
    EPPAS_ParameterMissingPaddingMode = 16,

    EPPAS_AttrMissingBias = 21,
    EPPAS_AttrMissingWeight = 22,
    EPPAS_AttrMissingRunningMean = 23,
    EPPAS_AttrMissingRunningVar = 24,
    EPPAS_AttrMissingOutFeatures = 25,
    EPPAS_AttrMissingYoloStrides = 26,
    EPPAS_AttrMissingYoloAnchorGrides = 27,
    EPPAS_AttrMissingYoloGrides = 28,

    EPPAS_ParameterAttrParseSuccess = 0
};

#endif //INFERFRAMEWORK_COMMON_HPP
