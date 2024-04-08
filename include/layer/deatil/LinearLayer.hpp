//
// Created by xyzzzh on 2024/4/7.
//

#ifndef INFERFRAMEWORK_LINEARLAYER_HPP
#define INFERFRAMEWORK_LINEARLAYER_HPP

#include "Common.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/abstract/NonParamLayer.hpp"
#include "layer/abstract/ParamLayer.hpp"

class LinearLayer : public ParamLayer {
public:
    explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

    EInferStatus forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                         std::vector<std::shared_ptr<Tensor>> &outputs) override;

    static EParseParameterAttrStatus get_instance(const std::shared_ptr<RuntimeOperator> &op,
                                                 std::shared_ptr<Layer> &linear_layer);

private:
    int32_t m_in_features = 0;
    int32_t m_out_features = 0;
    bool m_use_bias = false;
};


#endif //INFERFRAMEWORK_LINEARLAYER_HPP
