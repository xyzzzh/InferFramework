//
// Created by xyzzzh on 2024/4/3.
//

#ifndef INFERFRAMEWORK_MAXPOOLINGLAYER_HPP
#define INFERFRAMEWORK_MAXPOOLINGLAYER_HPP

#include "Common.hpp"
#include "layer/abstract/NonParamLayer.hpp"
#include "layer/abstract/LayerRegisterer.hpp"

class MaxPoolingLayer : public NonParamLayer {
public:
    MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w,
                    uint32_t pooling_size_h, uint32_t pooling_size_w,
                    uint32_t stride_h, uint32_t stride_w);

    EInferStatus forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                         std::vector<std::shared_ptr<Tensor>> &outputs) override;

    static EParseParameterAttrStatus
    get_instance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &max_layer);

private:
    uint32_t m_padding_h = 0;
    uint32_t m_padding_w = 0;
    uint32_t m_pooling_size_h = 0;
    uint32_t m_pooling_size_w = 0;
    uint32_t m_stride_h = 0;
    uint32_t m_stride_w = 0;
};


#endif //INFERFRAMEWORK_MAXPOOLINGLAYER_HPP
