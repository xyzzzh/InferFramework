//
// Created by xyzzzh on 2024/4/3.
//

#include "layer/deatil/ConvLayer.hpp"

ConvLayer::ConvLayer(uint32_t output_channel, uint32_t in_channel,
                     uint32_t kernel_h, uint32_t kernel_w,
                     uint32_t padding_h, uint32_t padding_w,
                     uint32_t stride_h, uint32_t stride_w,
                     uint32_t groups, bool use_bias)
        : ParamLayer("Convolution"),
          m_use_bias(use_bias),
          m_groups(groups),
          m_padding_h(padding_h),
          m_padding_w(padding_w),
          m_stride_h(stride_h),
          m_stride_w(stride_w) {
    if (groups != 1) {
        in_channel /= groups;
    }
    this->init_weight_param(output_channel, in_channel, kernel_h, kernel_w);
    if (m_use_bias) {
        this->init_bias_param(output_channel, 1, 1, 1);
    }
}

EInferStatus ConvLayer::forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs,
        std::vector<std::shared_ptr<Tensor>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the convolution layer is empty";
        return EInferStatus::EIS_InferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of the convolution "
                      "layer do not match";
        return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
    }

    if (this->m_weights.empty()) {
        LOG(ERROR) << "The number of kernel matrix in the convolution layer should "
                      "be greater than zero";
        return EInferStatus::EIS_InferFailedWeightParameterError;
    }

    if (this->m_use_bias && this->m_bias.size() != this->m_weights.size()) {
        LOG(ERROR) << "The number of kernel matrix and bias matrix do not match";
        return EInferStatus::EIS_InferFailedBiasParameterError;
    }

    if (!this->m_stride_h || !this->m_stride_w) {
        LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                      "greater than 0";
        return EInferStatus::EIS_InferFailedStrideParameterError;
    }

    const uint32_t kernel_count = this->m_weights.size();
    const uint32_t kernel_h = this->m_weights.at(0)->rows();
    const uint32_t kernel_w = this->m_weights.at(0)->cols();
    const uint32_t kernel_c = this->m_weights.at(0)->channels();
    const uint32_t row_len = kernel_h * kernel_w;
    CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
                    << "The size of kernel matrix in the convolution layer should be greater "
                       "than zero";

    for (uint32_t k = 0; k < kernel_count; ++k) {
        const std::shared_ptr<Tensor> &kernel = this->m_weights.at(k);
        CHECK(kernel->rows() == kernel_h);
        CHECK(kernel->cols() == kernel_w);
        CHECK(kernel->channels() == kernel_c);
    }
    const uint32_t kernel_count_group = kernel_count / this->m_groups;
    const uint32_t batch_size = inputs.size();

    if (this->m_kernel_matrix_arr.empty()) {
        this->init_IM2COL_weight();
    }

    if (!this->m_kernel_matrix_arr.empty()) {
        if (this->m_groups == 1) {
            CHECK(this->m_kernel_matrix_arr.size() == kernel_count_group)
                            << "The number of kernel matrix and kernel_count_group do not match";
        } else {
            CHECK(this->m_kernel_matrix_arr.size() == kernel_count)
                            << "The number of kernel matrix and kernel_count do not match";
        }
    }

    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor> &input = inputs.at(i);
        CHECK(input != nullptr && !input->empty())
                        << "The input tensor array in the convolution layer has an empty  "
                           "tensor "
                        << i << " th";

        const uint32_t input_c = input->channels();
        const uint32_t input_padded_h = input->rows() + 2 * this->m_padding_h;
        const uint32_t input_padded_w = input->cols() + 2 * this->m_padding_w;

        const uint32_t output_h =
                std::floor((int(input_padded_h) - int(kernel_h)) / this->m_stride_h + 1);
        const uint32_t output_w =
                std::floor((int(input_padded_w) - int(kernel_w)) / this->m_stride_w + 1);
        CHECK(output_h > 0 && output_w > 0)
                        << "The size of the output tensor should be greater than zero " << i
                        << " th";

        if (this->m_groups != 1) {
            CHECK(kernel_count % this->m_groups == 0);
            CHECK(input_c % this->m_groups == 0);
        }

        uint32_t col_len = output_h * output_w;
        CHECK(col_len > 0) << "Output_h x output_w for the convolution layer "
                              "should be greater than zero "
                           << i << " th";

        uint32_t input_c_group = input_c / this->m_groups;
        CHECK(input_c_group == kernel_c) << "The number of channel for the kernel "
                                            "matrix and input tensor do not match";

        for (uint32_t g = 0; g < this->m_groups; ++g) {
            const auto &input_matrix =
                    IM2COL(input, kernel_w, kernel_h, input->cols(), input->rows(),
                           input_c_group, g, row_len, col_len);
            std::shared_ptr<Tensor> output_tensor = outputs.at(i);
            if (output_tensor == nullptr || output_tensor->empty()) {
                output_tensor =
                        std::make_shared<Tensor>(kernel_count, output_h, output_w);
                outputs.at(i) = output_tensor;
            }

            CHECK(output_tensor->rows() == output_h &&
                  output_tensor->cols() == output_w &&
                  output_tensor->channels() == kernel_count)
                            << "The output tensor array in the convolution layer has an "
                               "incorrectly sized tensor "
                            << i << "th";

            const uint32_t kernel_count_group_start = kernel_count_group * g;
            for (uint32_t k = 0; k < kernel_count_group; ++k) {
                arma::frowvec kernel;
                if (this->m_groups == 1) {
                    kernel = this->m_kernel_matrix_arr.at(k);
                } else {
                    kernel = this->m_kernel_matrix_arr.at(kernel_count_group_start + k);
                }
                conv_GEMM_bias(input_matrix, output_tensor, g, k, kernel_count_group,
                               kernel, output_w, output_h);
            }
        }
    }
    return EInferStatus::EIS_InferSuccess;
}

arma::fmat ConvLayer::IM2COL(std::shared_ptr<Tensor> input, uint32_t kernel_w,
                             uint32_t kernel_h, uint32_t input_w,
                             uint32_t input_h, uint32_t input_c_group,
                             uint32_t group, uint32_t row_len,
                             uint32_t col_len) const {
    arma::fmat input_matrix(input_c_group * row_len, col_len);
    const uint32_t input_padded_h = input_h + 2 * this->m_padding_h;
    const uint32_t input_padded_w = input_w + 2 * this->m_padding_w;
    const float padding_value = 0.f;
    for (uint32_t ic = 0; ic < input_c_group; ++ic) {
        float *input_channel_ptr =
                input->matrix_raw_ptr(ic + group * input_c_group);
        uint32_t current_col = 0;
        uint32_t channel_row = ic * row_len;
        for (uint32_t w = 0; w < input_padded_w - kernel_w + 1; w += this->m_stride_w) {
            for (uint32_t r = 0; r < input_padded_h - kernel_h + 1; r += this->m_stride_h) {
                float *input_matrix_ptr =
                        input_matrix.colptr(current_col) + channel_row;
                current_col += 1;
                for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                    const uint32_t region_w = input_h * (w + kw - this->m_padding_w);
                    for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                        if ((kh + r >= this->m_padding_h && kw + w >= this->m_padding_w) &&
                            (kh + r < input_h + this->m_padding_h &&
                             kw + w < input_w + this->m_padding_w)) {
                            float *region_ptr =
                                    input_channel_ptr + region_w + (r + kh - this->m_padding_h);
                            *input_matrix_ptr = *region_ptr;
                        } else {
                            *input_matrix_ptr = padding_value;  // only support zero mode
                        }
                        input_matrix_ptr += 1;
                    }
                }
            }
        }
    }
    return input_matrix;
}

void ConvLayer::conv_GEMM_bias(
        const arma::fmat &input_matrix, std::shared_ptr<Tensor> output_tensor, uint32_t group,
        uint32_t kernel_index, uint32_t kernel_count_group,
        const arma::frowvec &kernel, uint32_t output_w, uint32_t output_h) const {
    arma::fmat output(
            output_tensor->matrix_raw_ptr(kernel_index + group * kernel_count_group),
            output_h, output_w, false, true);

    CHECK(output.size() == output_h * output_w)
                    << "Output_h x output_w for the convolution layer "
                       "should be output tensor size";

    if (!this->m_bias.empty() && this->m_use_bias) {
        std::shared_ptr<Tensor> bias;
        bias = this->m_bias.at(kernel_index);
        if (bias != nullptr && !bias->empty()) {
            float bias_value = bias->index(0);
            output = kernel * input_matrix + bias_value;
        } else {
            LOG(FATAL) << "Bias tensor is empty or nullptr";
        }
    } else {
        output = kernel * input_matrix;
    }
}

void ConvLayer::init_IM2COL_weight() {
    const uint32_t kernel_count = this->m_weights.size();
    CHECK(kernel_count > 0) << "kernel count must greater than zero";
    const uint32_t kernel_h = this->m_weights.at(0)->rows();
    const uint32_t kernel_w = this->m_weights.at(0)->cols();
    const uint32_t kernel_c = this->m_weights.at(0)->channels();
    const uint32_t row_len = kernel_h * kernel_w;
    CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
                    << "The size of kernel matrix should be greater than zero";

    for (uint32_t k = 0; k < kernel_count; ++k) {
        const std::shared_ptr<Tensor> &kernel = this->m_weights.at(k);
        CHECK(kernel->rows() == kernel_h);
        CHECK(kernel->cols() == kernel_w);
        CHECK(kernel->channels() == kernel_c);
    }

    if (this->m_groups == 1) {
        const uint32_t kernel_count_group = kernel_count / this->m_groups;
        std::vector<arma::frowvec> kernel_matrix_arr(kernel_count_group);
        arma::frowvec kernel_matrix_c(row_len * kernel_c);
        for (uint32_t k = 0; k < kernel_count_group; ++k) {
            const std::shared_ptr<Tensor> &kernel = this->m_weights.at(k);
            for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
                memcpy(kernel_matrix_c.memptr() + row_len * ic,
                       kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
            }
            kernel_matrix_arr.at(k) = kernel_matrix_c;
        }
        this->m_kernel_matrix_arr = std::move(kernel_matrix_arr);
    } else {
        // group != 1
        const uint32_t kernel_count_group = kernel_count / this->m_groups;
        std::vector<arma::frowvec> kernel_matrix_arr;
        for (uint32_t g = 0; g < this->m_groups; ++g) {
            arma::fmat kernel_matrix_c(1, row_len * kernel_c);
            for (uint32_t k = 0; k < kernel_count_group; ++k) {
                const std::shared_ptr<Tensor> &kernel =
                        this->m_weights.at(k + g * kernel_count_group);
                for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
                    memcpy(kernel_matrix_c.memptr() + row_len * ic,
                           kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
                }
                kernel_matrix_arr.emplace_back(kernel_matrix_c);
            }
        }
        CHECK(kernel_matrix_arr.size() == kernel_count);
        this->m_kernel_matrix_arr = std::move(kernel_matrix_arr);
    }
}

EParseParameterAttrStatus ConvLayer::get_instance(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &conv_layer) {
    CHECK(op != nullptr) << "Convolution operator is nullptr";
    const std::map<std::string, std::shared_ptr<RuntimeParameter>> &params =
            op->m_params;

    if (params.find("dilation") == params.end()) {
        LOG(ERROR) << "Can not find the dilation parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingDilation;
    }

    auto dilation_param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
            params.at("dilation"));

    if (dilation_param == nullptr || dilation_param->value.size() != 2) {
        LOG(ERROR) << "Can not find the dilation parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingDilation;
    }

    CHECK(dilation_param->value.at(0) != 1 || dilation_param->value.at(1))
                    << "Only support dilation value equals to one!";

    if (params.find("in_channels") == params.end()) {
        LOG(ERROR) << "Can not find the in channel parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingInChannel;
    }
    auto in_channel =
            std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("in_channels"));
    if (!in_channel) {
        LOG(ERROR) << "Can not find the in channel parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingInChannel;
    }

    if (params.find("out_channels") == params.end()) {
        LOG(ERROR) << "Can not find the out channel parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingOutChannel;
    }

    auto out_channel =
            std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("out_channels"));
    if (!out_channel) {
        LOG(ERROR) << "Can not find the out channel parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingOutChannel;
    }

    if (params.find("padding") == params.end()) {
        LOG(ERROR) << "Can not find the padding parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingPadding;
    }

    auto padding =
            std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
    if (!padding) {
        LOG(ERROR) << "Can not find the padding parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingPadding;
    }

    if (params.find("bias") == params.end()) {
        LOG(ERROR) << "Can not find the bias parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingUseBias;
    }
    auto use_bias =
            std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
    if (!use_bias) {
        LOG(ERROR) << "Can not find the bias parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingUseBias;
    }

    if (params.find("stride") == params.end()) {
        LOG(ERROR) << "Can not find the stride parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingStride;
    }
    auto stride =
            std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
    if (!stride) {
        LOG(ERROR) << "Can not find the stride parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingStride;
    }

    if (params.find("kernel_size") == params.end()) {
        LOG(ERROR) << "Can not find the kernel parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingKernel;
    }
    auto kernel = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
            params.at("kernel_size"));
    if (!kernel) {
        LOG(ERROR) << "Can not find the kernel parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingKernel;
    }

    if (params.find("padding_mode") != params.end()) {
        auto padding_mode = std::dynamic_pointer_cast<RuntimeParameterString>(
                params.at("padding_mode"));
        if (padding_mode == nullptr) {
            LOG(ERROR) << "Can not find the padding parameter";
            return EParseParameterAttrStatus::EPPAS_ParameterMissingPaddingMode;
        } else {
            const std::string &padding_mode_str = padding_mode->value;
            if (padding_mode_str != "zeros") {
                LOG(ERROR) << "Padding mode unsupported: " << padding_mode_str;
                return EParseParameterAttrStatus::EPPAS_ParameterMissingPaddingMode;
            }
        }
    } else {
        LOG(ERROR) << "Can not find the padding parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingPaddingMode;
    }

    auto groups =
            std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("groups"));
    if (!groups) {
        LOG(ERROR) << "Can not find the groups parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingGroups;
    }

    const uint32_t dims = 2;
    const std::vector<int> &kernels = kernel->value;
    const std::vector<int> &paddings = padding->value;
    const std::vector<int> &strides = stride->value;
    if (paddings.size() != dims) {
        LOG(ERROR) << "Can not find the right padding parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingPadding;
    }

    if (strides.size() != dims) {
        LOG(ERROR) << "Can not find the right stride parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingStride;
    }

    if (kernels.size() != dims) {
        LOG(ERROR) << "Can not find the right kernel size parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingKernel;
    }

    // kernel的方向是倒置的
    conv_layer = std::make_shared<ConvLayer>(
            out_channel->value, in_channel->value, kernels.at(0), kernels.at(1),
            paddings.at(0), paddings.at(1), strides.at(0), strides.at(1),
            groups->value, use_bias->value);

    // load weights
    const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &attrs =
            op->m_attribute;
    if (use_bias->value) {
        if (attrs.find("bias") == attrs.end()) {
            LOG(ERROR) << "Can not find the bias attribute";
            return EParseParameterAttrStatus::EPPAS_AttrMissingBias;
        }
        const auto &bias = attrs.at("bias");
        const std::vector<uint32_t> &bias_shape = bias->m_shapes;
        if (bias_shape.empty() || bias_shape.at(0) != out_channel->value) {
            LOG(ERROR) << "The attribute of bias shape is wrong";
            return EParseParameterAttrStatus::EPPAS_AttrMissingBias;
        }

        const std::vector<float> &bias_values = bias->get_weight_data<float>();
        conv_layer->set_bias(bias_values);
    }

    if (attrs.find("weight") == attrs.end()) {
        LOG(ERROR) << "Can not find the weight attribute";
        return EParseParameterAttrStatus::EPPAS_AttrMissingWeight;
    }

    const auto &weight = attrs.at("weight");
    const std::vector<uint32_t> &weight_shape = weight->m_shapes;
    if (weight_shape.empty()) {
        LOG(ERROR) << "The attribute of weight shape is wrong";
        return EParseParameterAttrStatus::EPPAS_AttrMissingWeight;
    }

    const std::vector<float> &weight_values = weight->get_weight_data<float>();
    conv_layer->set_weights(weight_values);

    auto conv_layer_derived =
            std::dynamic_pointer_cast<ConvLayer>(conv_layer);
    CHECK(conv_layer_derived != nullptr);
    conv_layer_derived->init_IM2COL_weight();
    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;
}

LayerRegistererWrapper conv_get_instance("nn.Conv2d", ConvLayer::get_instance);