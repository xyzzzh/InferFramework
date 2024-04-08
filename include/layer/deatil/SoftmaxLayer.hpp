//
// Created by xyzzzh on 2024/4/7.
//

#ifndef INFERFRAMEWORK_SOFTMAXLAYER_HPP
#define INFERFRAMEWORK_SOFTMAXLAYER_HPP

#include "Common.hpp"
#include "fmath.hpp"
#include "layer/abstract/NonParamLayer.hpp"

class SoftmaxLayer : public NonParamLayer {
public:
    explicit SoftmaxLayer(int dim = -1);

    EInferStatus forward(
            const std::vector<std::shared_ptr<Tensor>> &inputs,
            std::vector<std::shared_ptr<Tensor>> &outputs) override;

    static EParseParameterAttrStatus create_instance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &softmax_layer);

private:
    int m_softmax_dim = -1;
};


#endif //INFERFRAMEWORK_SOFTMAXLAYER_HPP
