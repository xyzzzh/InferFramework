//
// Created by xyzzzh on 2024/4/2.
//

#ifndef INFERFRAMEWORK_RELULAYER_HPP
#define INFERFRAMEWORK_RELULAYER_HPP

#include "Common.hpp"
#include "layer/abstract/NonParamLayer.hpp"

class ReluLayer : public NonParamLayer {
public:
    ReluLayer() : NonParamLayer("Relu") {}

    EInferStatus forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                         std::vector<std::shared_ptr<Tensor>> &outputs) override;

    // Relu初始化
    static EParseParameterAttrStatus
    get_instance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &relu_layer);

};


#endif //INFERFRAMEWORK_RELULAYER_HPP
