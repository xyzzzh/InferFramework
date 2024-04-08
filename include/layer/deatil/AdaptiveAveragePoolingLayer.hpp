//
// Created by xyzzzh on 2024/4/7.
//

#ifndef INFERFRAMEWORK_ADAPTIVEAVERAGEPOOLINGLAYER_HPP
#define INFERFRAMEWORK_ADAPTIVEAVERAGEPOOLINGLAYER_HPP

#include "Common.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/abstract/NonParamLayer.hpp"
#include "layer/abstract/ParamLayer.hpp"

class AdaptiveAveragePoolingLayer : public NonParamLayer {
public:
    explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w);

    EInferStatus forward(
            const std::vector<std::shared_ptr<Tensor>> &inputs,
            std::vector<std::shared_ptr<Tensor>> &outputs) override;

    static EParseParameterAttrStatus create_instance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &avg_layer);

private:
    uint32_t m_output_h = 0;
    uint32_t m_output_w = 0;
};


#endif //INFERFRAMEWORK_ADAPTIVEAVERAGEPOOLINGLAYER_HPP
