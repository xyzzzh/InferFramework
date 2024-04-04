//
// Created by xyzzzh on 2024/4/3.
//

#include "layer/deatil/MaxPoolingLayer.hpp"

MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h,
                                 uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w) :
        NonParamLayer("MaxPooling"),
        m_padding_h(padding_h),
        m_padding_w(padding_w),
        m_pooling_size_h(pooling_size_h),
        m_pooling_size_w(pooling_size_w),
        m_stride_h(stride_h),
        m_stride_w(stride_w) {}

EInferStatus MaxPoolingLayer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                      std::vector<std::shared_ptr<Tensor>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
        return EInferStatus::EIS_InferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR)
                << "The input and output tensor array size of the max pooling layer "
                   "do not match";
        return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
    }

    const uint32_t batch = inputs.size();
    const uint32_t pooling_h = this->m_pooling_size_h;
    const uint32_t pooling_w = this->m_pooling_size_w;

    if (pooling_h == 0 || pooling_w == 0) {
        LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                      "greater than 0";
        return EInferStatus::EIS_InferFailedStrideParameterError;
    }

    for (uint32_t i = 0; i < batch; i++) {
        const std::shared_ptr<Tensor> &input_data = inputs[i];
        if (input_data == nullptr || input_data->empty()) {
            LOG(ERROR) << "The input tensor array in the max pooling layer has an " << i << "th";
            return EInferStatus::EIS_InferFailedInputEmpty;
        }
        uint32_t input_h = input_data->rows();
        uint32_t input_w = input_data->cols();
        uint32_t output_h = uint32_t(std::floor(
                ((int(input_h) - int(pooling_h) + 2 * this->m_padding_h) / (m_stride_h)) + 1
        ));
        uint32_t output_w = uint32_t(std::floor(
                ((int(input_w) - int(pooling_w) + 2 * this->m_padding_w) / (m_stride_w)) + 1
        ));

        if (output_w == 0 || output_h == 0) {
            LOG(ERROR) << "The output size of tensor " << i << "th" << " in the max pooling layer is less than zero";
            return EInferStatus::EIS_InferFailedOutputSizeError;
        }

        const std::shared_ptr<Tensor> &output_data = outputs[i];
        if (output_data != nullptr && !output_data->empty()) {
            if (output_data->rows() != output_h ||
                output_data->cols() != output_w) {
                LOG(ERROR) << "The output tensor array in the max pooling layer has an incorrectly sized tensor "
                           << i << "th";
                return EInferStatus::EIS_InferFailedOutputSizeError;
            }
        }
    }

    for (uint32_t i = 0; i < batch; i++) {
        const std::shared_ptr<Tensor> &input_data = inputs[i];
        if (input_data == nullptr || input_data->empty()) {
            LOG(ERROR) << "The input tensor array in the max pooling layer has an empty tensor " << i << "th";
        }

        const uint32_t input_h = input_data->rows();
        const uint32_t input_w = input_data->cols();
        const uint32_t input_padded_h = input_data->rows() + 2 * this->m_padding_h;
        const uint32_t input_padded_w = input_data->cols() + 2 * this->m_padding_w;

        const uint32_t input_ch = input_data->channels();
        const uint32_t output_h = uint32_t(std::floor((int(input_padded_h) - int(pooling_h)) / this->m_stride_h + 1));
        const uint32_t output_w = uint32_t(std::floor((int(input_padded_w) - int(pooling_w)) / this->m_stride_w + 1));

        std::shared_ptr<Tensor> output_data = outputs[i];
        if (output_data == nullptr || output_data->empty()) {
            output_data = std::make_shared<Tensor>(input_ch, output_h, output_w);
            outputs[i] = output_data;
        }

        CHECK(output_data->rows() == output_h && output_data->cols() == output_w &&
              output_data->channels() == input_ch)
                        << "The output tensor array in the max pooling layer has an incorrectly sized tensor " << i
                        << "th";

        for (uint32_t ic = 0; ic < input_ch; ic++) {
            const arma::fmat &input_channel = input_data->slice(ic);
            arma::fmat &output_channel = output_data->slice(ic);
            for (uint32_t c = 0; c < input_padded_w - pooling_w + 1; c += this->m_stride_w) {
                int output_col = int(c / this->m_stride_w);
                for (uint32_t r = 0; r < input_padded_h - pooling_h + 1;
                     r += this->m_stride_h) {
                    int output_row = int(r / this->m_stride_h);
                    float *output_channel_ptr = output_channel.colptr(output_col);
                    float max_value = std::numeric_limits<float>::lowest();
                    for (uint32_t w = 0; w < pooling_w; ++w) {
                        const float *col_ptr = input_channel.colptr(c + w - this->m_padding_w);
                        for (uint32_t h = 0; h < pooling_h; ++h) {
                            float current_value = 0.f;
                            if ((h + r >= this->m_padding_h && w + c >= this->m_padding_w) &&
                                (h + r < input_h + this->m_padding_h &&
                                 w + c < input_w + this->m_padding_w)) {
                                current_value = *(col_ptr + r + h - this->m_padding_h);
                            } else {
                                current_value = std::numeric_limits<float>::lowest();
                            }
                            max_value = max_value > current_value ? max_value : current_value;
                        }
                    }
                    *(output_channel_ptr + output_row) = max_value;
                }
            }
        }
    }

}

EParseParameterAttrStatus
MaxPoolingLayer::get_instance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &max_layer) {
    CHECK(op != nullptr) << "MaxPooling get instance failed, operator is nullptr";
    const std::map<std::string, std::shared_ptr<RuntimeParameter>> &params = op->m_params;
    if (params.find("stride") == params.end()) {
        LOG(ERROR) << "Can not find the stride parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingStride;
    }

    auto stride = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
    if (stride == nullptr) {
        LOG(ERROR) << "Can not find the stride parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingStride;
    }

    if (params.find("padding") == params.end()) {
        LOG(ERROR) << "Can not find the padding parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingPadding;
    }

    auto padding = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
    if (padding == nullptr) {
        LOG(ERROR) << "Can not find the padding parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingPadding;
    }

    if (params.find("kernel_size") == params.end()) {
        LOG(ERROR) << "Can not find the kernel size parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingKernel;
    }

    auto kernel_size = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("kernel_size"));
    if (!kernel_size) {
        LOG(ERROR) << "Can not find the kernel size parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingKernel;
    }

    const auto &stride_values = stride->value;
    const auto &padding_values = padding->value;
    const auto &kernel_size_values = kernel_size->value;

    const uint32_t dims = 2;
    if (padding_values.size() != dims) {
        LOG(ERROR) << "Can not find the right padding parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingPadding;
    }

    if (stride_values.size() != dims) {
        LOG(ERROR) << "Can not find the right stride parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingStride;
    }

    if (kernel_size_values.size() != dims) {
        LOG(ERROR) << "Can not find the right kernel size parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingKernel;
    }

    // MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w,
    //                    uint32_t pooling_size_h, uint32_t pooling_size_w,
    //                    uint32_t stride_h, uint32_t stride_w);
    max_layer = std::make_shared<MaxPoolingLayer>(
            padding_values[0], padding_values[1],
            kernel_size_values[0], kernel_size_values[1],
            stride_values[0], stride_values[1]);

    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;
}

LayerRegistererWrapper maxpooling_get_instance("nn.MaxPool2d", MaxPoolingLayer::get_instance);
