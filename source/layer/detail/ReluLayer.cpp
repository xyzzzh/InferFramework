//
// Created by xyzzzh on 2024/4/2.
//

#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/deatil/ReluLayer.hpp"

EInferStatus
ReluLayer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the relu layer is empty";
        return EInferStatus::EIS_InferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of the relu layer do "
                      "not match";
        return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
    }

    const uint32_t batch_size = inputs.size();
    for (uint32_t i = 0; i < batch_size; i++) {
        const std::shared_ptr<Tensor> &input_data = inputs[i];
        const std::shared_ptr<Tensor> &output_data = outputs[i];
        if (input_data == nullptr || input_data->empty()) {
            LOG(ERROR)
                    << "The input tensor array in the relu layer has an empty tensor "
                    << i << " th";
            return EInferStatus::EIS_InferFailedInputEmpty;
        }
        if (output_data != nullptr && !output_data->empty()) {
            if (input_data->shapes() != output_data->shapes()) {
                LOG(ERROR) << "The input and output tensor shapes of the relu "
                              "layer do not match "
                           << i << " th";
                return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
            }
        }
    }

    for (uint32_t i = 0; i < batch_size; i++) {
        const std::shared_ptr<Tensor> &input = inputs[i];
        CHECK(input == nullptr || !input->empty())
                        << "The input tensor array in the relu layer has an empty tensor " << i << " th";
        std::shared_ptr<Tensor> output = outputs[i];
        if (output == nullptr || output->empty()) {
            LOG(ERROR) << "The output tensor array in the relu layer has an empty tensor " << i << " th";
            output = std::make_shared<Tensor>(input->shapes());
            outputs[i] = output;
        }
        CHECK(output->shapes() == input->shapes())
                        << "The input and output tensor shapes of the relu layer do not match " << i << " th";
        for (uint32_t j = 0; j < input->size(); j++) {
            float value = input->index(j);
            output->index(j) = value > 0.f ? value : 0;
        }
    }
    return EInferStatus::EIS_InferSuccess;
}

EParseParameterAttrStatus
ReluLayer::get_instance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &relu_layer) {
    CHECK(op != nullptr) << "ReLU operator is nullptr";
    relu_layer = std::make_shared<ReluLayer>();
    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;
}

// 使用工具类注册算子
LayerRegistererWrapper relu_get_instance("nn.ReLU", ReluLayer::get_instance);