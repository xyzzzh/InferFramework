//
// Created by xyzzzh on 2024/4/7.
//

#include "layer/deatil/SoftmaxLayer.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "fmath.hpp"
#include "Utils.hpp"


SoftmaxLayer::SoftmaxLayer(int dim)
        : NonParamLayer("Softmax"), m_softmax_dim(dim) {}

EInferStatus SoftmaxLayer::forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs,
        std::vector<std::shared_ptr<Tensor>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the softmax layer is empty";
        return EInferStatus::EIS_InferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of the softmax layer "
                      "do not match";
        return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
    }

    const uint32_t batch_size = inputs.size();
    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor> &input = inputs.at(i);
        CHECK(input != nullptr && !input->empty())
                        << "The input tensor array in the softmax layer has an empty tensor "
                        << i << " th";

        std::shared_ptr<Tensor> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            output = std::make_shared<Tensor>(input->shapes());
            outputs.at(i) = output;
        }
        CHECK(input->shapes() == output->shapes())
                        << "The input and output tensor shapes of the softmax layer do not "
                           "match "
                        << i << " th";
        int dim = this->m_softmax_dim;
        std::vector<uint32_t> raw_shapes = input->raw_shapes();

        if (dim < 0) {
            dim += int(raw_shapes.size());
        }

        if (dim < 0 || dim >= 3 || dim > raw_shapes.size()) {
            LOG(FATAL) << "Error softmax dimension, which need between 0 and 2, "
                          "but dimension is "
                       << dim;
        }
        const uint32_t padding_size_num = 3 - raw_shapes.size();
        for (uint32_t j = 0; j < padding_size_num; ++j) {
            raw_shapes.push_back(1);
        }

        /**
         * [...(inner size) dim ...(outer_size)
         * 将输入的数据按dim维度拆分为两部分，分别为inner和outer
         * 开始位置到dim轴位置的数据量是inner_size,
         * dim轴位置到结束位置的数据量是outer_sizes
         */
        const uint32_t inner_sizes = std::accumulate(
                raw_shapes.begin() + dim + 1, raw_shapes.end(), 1, std::multiplies());
        const uint32_t outer_sizes = std::accumulate(
                raw_shapes.begin(), raw_shapes.begin() + dim, 1, std::multiplies());

        // dim轴数据的数量
        const uint32_t axis_sizes = raw_shapes.at(dim);
        CHECK_EQ(axis_sizes * outer_sizes * inner_sizes, input->size());

        const auto &input_values = input->values(true);
        std::vector<float> output_values(input_values.size());
        for (uint32_t outer_size = 0; outer_size < outer_sizes; ++outer_size) {
            for (uint32_t inner_size = 0; inner_size < inner_sizes; ++inner_size) {
                float max_value = std::numeric_limits<float>::lowest();
                // 迭代当前dim中的数据，并找到其中的最大值
                for (uint32_t axis_size = 0; axis_size < axis_sizes; ++axis_size) {
                    uint32_t index = get_pos_index(outer_size, inner_size, axis_size, axis_sizes, inner_sizes);
                    float cur_value = input_values.at(index);
                    if (cur_value > max_value) {
                        max_value = cur_value;
                    }
                }

                float sum_value = 0.f;
                // 迭代当前dim中的数据，并进行求和
                for (uint32_t axis_size = 0; axis_size < axis_sizes; ++axis_size) {
                    uint32_t index = get_pos_index(outer_size, inner_size, axis_size, axis_sizes, inner_sizes);
                    float cur_value = input_values.at(index);
                    float exp_sub_value = fmath::exp(cur_value - max_value);

                    sum_value += exp_sub_value;
                    output_values.at(index) = exp_sub_value;
                }

                // 迭代当前dim中的数据，求exp(cur_value - max_value) / sum_value
                for (uint32_t axis_size = 0; axis_size < axis_sizes; ++axis_size) {
                    uint32_t index = get_pos_index(outer_size, inner_size, axis_size, axis_sizes, inner_sizes);

                    float exp_sub_value = output_values.at(index);
                    output_values.at(index) = exp_sub_value / sum_value;
                }
            }
        }
        output->fill(output_values, true);
    }
    return EInferStatus::EIS_InferSuccess;
}

EParseParameterAttrStatus SoftmaxLayer::create_instance(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &softmax_layer) {
    CHECK(op != nullptr) << "SoftMax operator is nullptr";
    const auto &params = op->m_params;
    if (params.find("dim") == params.end()) {
        return EParseParameterAttrStatus::EPPAS_ParameterMissingDim;
    }

    auto dim_param = params.at("dim");
    if (dim_param == nullptr) {
        return EParseParameterAttrStatus::EPPAS_ParameterMissingDim;
    }

    auto dim = std::dynamic_pointer_cast<RuntimeParameterInt>(dim_param);
    if (dim == nullptr) {
        return EParseParameterAttrStatus::EPPAS_ParameterMissingDim;
    }
    softmax_layer = std::make_shared<SoftmaxLayer>(dim->value);  // 创建softmax层
    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;
}

LayerRegistererWrapper softmax_create_instanceNN("nn.Softmax", SoftmaxLayer::create_instance);
LayerRegistererWrapper softmax_create_instanceF("F.softmax", SoftmaxLayer::create_instance);