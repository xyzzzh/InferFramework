//
// Created by xyzzzh on 2024/4/3.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "Common.hpp"
#include "runtime/RuntimeParameter.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/deatil/MaxPoolingLayer.hpp"

TEST(test_maxpooling, create_layer_poolingforward) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->m_type = "nn.MaxPool2d";
    std::vector<int> strides{2, 2};
    std::shared_ptr<RuntimeParameter> stride_param = std::make_shared<RuntimeParameterIntArray>(strides);
    op->m_params.insert({"stride", stride_param});

    std::vector<int> kernel{2, 2};
    std::shared_ptr<RuntimeParameter> kernel_param = std::make_shared<RuntimeParameterIntArray>(strides);
    op->m_params.insert({"kernel_size", kernel_param});

    std::vector<int> paddings{0, 0};
    std::shared_ptr<RuntimeParameter> padding_param = std::make_shared<RuntimeParameterIntArray>(paddings);
    op->m_params.insert({"padding", padding_param});

    std::shared_ptr<Layer> layer;
    layer = LayerRegisterer::create_layer(op);
    ASSERT_NE(layer, nullptr);
}

TEST(test_maxpooling, create_layer_poolingforward_1) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->m_type = "nn.MaxPool2d";
    std::vector<int> strides{2, 2};
    std::shared_ptr<RuntimeParameter> stride_param = std::make_shared<RuntimeParameterIntArray>(strides);
    op->m_params.insert({"stride", stride_param});

    std::vector<int> kernel{2, 2};
    std::shared_ptr<RuntimeParameter> kernel_param = std::make_shared<RuntimeParameterIntArray>(strides);
    op->m_params.insert({"kernel_size", kernel_param});

    std::vector<int> paddings{0, 0};
    std::shared_ptr<RuntimeParameter> padding_param = std::make_shared<RuntimeParameterIntArray>(paddings);
    op->m_params.insert({"padding", padding_param});

    std::shared_ptr<Layer> layer;
    layer = LayerRegisterer::create_layer(op);
    ASSERT_NE(layer, nullptr);

    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(1, 4, 4);
    arma::fmat input = arma::fmat("1,2,3,4;"
                                  "2,3,4,5;"
                                  "3,4,5,6;"
                                  "4,5,6,7");
    tensor->data().slice(0) = input;
    std::vector<std::shared_ptr<Tensor>> inputs(1);
    inputs.at(0) = tensor;
    std::vector<std::shared_ptr<Tensor>> outputs(1);
    layer->forward(inputs, outputs);

    ASSERT_EQ(outputs.size(), 1);
    outputs.front()->show();
}