//
// Created by xyzzzh on 2024/4/7.
//

#include "layer/deatil/AdaptiveAveragePoolingLayer.hpp"

AdaptiveAveragePoolingLayer::AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w)
        : NonParamLayer("AdaptiveAveragePooling"),
          m_output_h(output_h),
          m_output_w(output_w) {
    CHECK_GT(m_output_h, 0);
    CHECK_GT(m_output_w, 0);
}

EInferStatus AdaptiveAveragePoolingLayer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                                  std::vector<std::shared_ptr<Tensor>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR)
                << "The input tensor array in the adaptive pooling layer is empty";
        return EInferStatus::EIS_InferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of the adaptive "
                      "pooling layer "
                      "do not match";
        return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
    }

    const uint32_t batch = inputs.size();
    for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<Tensor> &input_data = inputs.at(i);
        const std::shared_ptr<Tensor> &output_data = outputs.at(i);
        if (input_data == nullptr || input_data->empty()) {
            LOG(ERROR) << "The input tensor array in the adaptive pooling layer has "
                          "an empty tensor "
                       << i << "th";
            return EInferStatus::EIS_InferFailedInputEmpty;
        }
        if (output_data != nullptr && !output_data->empty()) {
            if (output_data->rows() != this->m_output_h ||
                output_data->cols() != this->m_output_w) {
                LOG(ERROR) << "The output tensor array in the adaptive pooling layer "
                              "has an incorrectly sized tensor "
                           << i << "th";
                return EInferStatus::EIS_InferFailedOutputSizeError;
            }
        }
    }

    for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<Tensor> &input_data = inputs.at(i);
        CHECK(input_data != nullptr && !input_data->empty())
                        << "The input tensor array in the adaptive pooling layer has an empty "
                           "tensor "
                        << i << "th";

        const uint32_t input_h = input_data->rows();
        const uint32_t input_w = input_data->cols();
        const uint32_t input_c = input_data->channels();
        const uint32_t stride_h = uint32_t(std::floor(input_h / this->m_output_h));
        const uint32_t stride_w = uint32_t(std::floor(input_w / this->m_output_w));
        CHECK(stride_w > 0 && stride_h > 0)
                        << "The stride parameter is set incorrectly. It must always be greater "
                           "than 0";

        const uint32_t pooling_h =
        (int) input_h - (int(this->m_output_h) -1) *int(stride_h);
        const uint32_t pooling_w =
        (int) input_w - (int(this->m_output_w) -1) *int(stride_w);
        CHECK(pooling_w > 0 && pooling_h > 0)
                        << "The pooling parameter is set incorrectly. It must always be "
                           "greater than 0";

        std::shared_ptr<Tensor> output_data = outputs.at(i);
        if (output_data == nullptr || output_data->empty()) {
            DLOG(ERROR) << "The output tensor array in the adaptive pooling layer "
                           "has an empty tensor "
                        << i << "th";
            output_data =
                    std::make_shared<Tensor>(input_c, this->m_output_h, this->m_output_w);
            outputs.at(i) = output_data;
        }

        CHECK(output_data->rows() == this->m_output_h &&
              output_data->cols() == this->m_output_w &&
              output_data->channels() == input_c)
                        << "The output tensor array in the adaptive pooling layer has an "
                           "incorrectly sized tensor "
                        << i << "th";

        const uint32_t pooling_size = pooling_h * pooling_w;
        for (uint32_t ic = 0; ic < input_c; ++ic) {
            const arma::fmat &input_channel = input_data->slice(ic);
            arma::fmat &output_channel = output_data->slice(ic);
            for (uint32_t c = 0; c < input_w - pooling_w + 1; c += stride_w) {
                int output_col = int(c / stride_w);
                for (uint32_t r = 0; r < input_h - pooling_h + 1; r += stride_h) {
                    int output_row = int(r / stride_h);
                    float mean_value = 0.f;
                    float *output_channel_ptr = output_channel.colptr(output_col);
                    for (uint32_t w = 0; w < pooling_w; ++w) {
                        const float *col_ptr = input_channel.colptr(c + w) + r;
                        for (uint32_t h = 0; h < pooling_h; ++h) {
                            float current_value = *(col_ptr + h);
                            mean_value = mean_value + current_value;
                        }
                    }
                    *(output_channel_ptr + output_row) = mean_value / float(pooling_size);
                }
            }
        }
    }
    return EInferStatus::EIS_InferSuccess;
}

EParseParameterAttrStatus AdaptiveAveragePoolingLayer::create_instance(const std::shared_ptr<RuntimeOperator> &op,
                                                                       std::shared_ptr<Layer> &avg_layer) {
    CHECK(op != nullptr) << "Adaptive pooling operator is nullptr";
    const auto &params = op->m_params;
    CHECK(!params.empty()) << "Operator parameter is empty";

    auto output_hw = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
            params.at("output_size"));
    if (!output_hw) {
        LOG(ERROR) << "Can not find the output size parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingOutHW;
    }

    const auto &output_hw_arr = output_hw->value;
    if (output_hw_arr.size() != 2) {
        LOG(ERROR) << "Can not find the output size parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingOutHW;
    }
    avg_layer = std::make_shared<AdaptiveAveragePoolingLayer>(
            output_hw_arr.at(0), output_hw_arr.at(1));
    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;
}

LayerRegistererWrapper adaptive_avgpooling_create_instance("nn.AdaptiveAvgPool2d",
                                                           AdaptiveAveragePoolingLayer::create_instance);