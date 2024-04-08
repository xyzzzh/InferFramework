//
// Created by xyzzzh on 2024/4/7.
//

#ifndef INFERFRAMEWORK_FLATTENLAYER_HPP
#define INFERFRAMEWORK_FLATTENLAYER_HPP

#include "Common.hpp"
#include "Utils.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/abstract/NonParamLayer.hpp"
#include "layer/abstract/ParamLayer.hpp"

class FlattenLayer : public NonParamLayer {
public:
    explicit FlattenLayer(int start_dim, int end_dim);

    EInferStatus forward(
            const std::vector<std::shared_ptr<Tensor>> &inputs,
            std::vector<std::shared_ptr<Tensor>> &outputs) override;

    static EParseParameterAttrStatus create_instance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &flatten_layer);

private:
    int m_start_dim = 0;
    int m_end_dim = 0;
};


#endif //INFERFRAMEWORK_FLATTENLAYER_HPP
