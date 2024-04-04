//
// Created by xyzzzh on 2024/4/4.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "Common.hpp"
#include "runtime/RuntimeParameter.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/deatil/ConvLayer.hpp"


TEST(test_registry, create_layer_convforward) {
    const uint32_t batch_size = 1;
    std::vector<std::shared_ptr<Tensor>> inputs(batch_size);
    std::vector<std::shared_ptr<Tensor>> outputs(batch_size);

    const uint32_t in_channel = 2;
    for (uint32_t i = 0; i < batch_size; ++i) {
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(in_channel, 4, 4);
        input->data().slice(0) = "1,2,3,4;"
                                 "5,6,7,8;"
                                 "9,10,11,12;"
                                 "13,14,15,16;";

        input->data().slice(1) = "1,2,3,4;"
                                 "5,6,7,8;"
                                 "9,10,11,12;"
                                 "13,14,15,16;";
        inputs.at(i) = input;
    }
    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    std::vector<std::shared_ptr<Tensor>> weights;
    for (uint32_t i = 0; i < kernel_count; ++i) {
        std::shared_ptr<Tensor> kernel = std::make_shared<Tensor>(in_channel, kernel_h, kernel_w);
        kernel->data().slice(0) = arma::fmat("1,2,3;"
                                             "3,2,1;"
                                             "1,2,3;");
        kernel->data().slice(1) = arma::fmat("1,2,3;"
                                             "3,2,1;"
                                             "1,2,3;");
        weights.push_back(kernel);
    }
    ConvLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                         0, stride_h, stride_w, 1, false);
    conv_layer.set_weights(weights);
    conv_layer.forward(inputs, outputs);
    outputs.at(0)->show();
}