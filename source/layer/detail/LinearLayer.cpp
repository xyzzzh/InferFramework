//
// Created by xyzzzh on 2024/4/7.
//

#include "layer/deatil/LinearLayer.hpp"

LinearLayer::LinearLayer(int32_t in_features, int32_t out_features, bool use_bias) :
        ParamLayer("Linear"),
        m_use_bias(use_bias),
        m_in_features(in_features),
        m_out_features(out_features) {
    CHECK_GT(in_features, 0);
    CHECK_GT(out_features, 0);
    this->init_weight_param(1, 1, out_features, in_features);
    if (use_bias) {
        this->init_bias_param(1, 1, 1, out_features);
    }
}

EInferStatus LinearLayer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                  std::vector<std::shared_ptr<Tensor>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the linear layer is empty";
        return EInferStatus::EIS_InferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of linear layer do "
                      "not match";
        return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
    }

    if (this->m_weights.empty()) {
        LOG(ERROR) << "The weight tensor in the linear layer is empty";
        return EInferStatus::EIS_InferFailedWeightParameterError;
    } else {
        if (this->m_use_bias && this->m_weights.size() != this->m_bias.size()) {
            LOG(ERROR) << "The size of the weight and bias tensor do not match";
            return EInferStatus::EIS_InferFailedBiasParameterError;
        }
    }

    if (this->m_weights.size() != 1) {
        LOG(ERROR) << "Need one weight tensor in the linear layer";
        return EInferStatus::EIS_InferFailedWeightParameterError;
    }

    if (this->m_use_bias && this->m_bias.size() != 1) {
        LOG(ERROR) << "Need one bias tensor in the linear layer";
        return EInferStatus::EIS_InferFailedBiasParameterError;
    }

    uint32_t batch = inputs.size();
    const std::shared_ptr<Tensor> &weight = this->m_weights.front();
    arma::fmat weight_data(weight->raw_ptr(), this->m_out_features, this->m_in_features,
                           false, true);
    const arma::fmat &weight_data_t = weight_data.t();

    for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<Tensor> &input = inputs.at(i);
        CHECK(input != nullptr && !input->empty())
                        << "The input tensor array in the linear layer has an empty tensor "
                        << i << " th";
        const std::vector<uint32_t> &input_shapes = input->shapes();

        const uint32_t feature_dims = input_shapes.at(1);
        const uint32_t in_features = input_shapes.at(2);
        CHECK(weight_data.n_rows == this->m_out_features)
                        << "The row of weight tensor should be same to output_features_";
        CHECK(weight_data.n_cols == in_features && in_features == this->m_in_features)
                        << "The col of weight tensor should be same to input_features_";

        arma::fmat input_vec((float *) input->raw_ptr(), feature_dims, this->m_in_features,
                             false, true);

        std::shared_ptr<Tensor> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            output = std::make_shared<Tensor>(1, this->m_out_features, feature_dims);
            outputs.at(i) = output;
        }
        CHECK(output->channels() == 1 && output->rows() == feature_dims &&
              output->cols() == this->m_out_features)
                        << "The row of output tensor should be same to feature_dims_ and the "
                           "col of output tensor should be same to output_features_ "
                        << i << " th";
        const auto &output_raw_shapes = output->raw_shapes();
        if (output_raw_shapes.size() == 2) {
            CHECK(output_raw_shapes.at(0) == feature_dims &&
                  output_raw_shapes.at(1) == this->m_out_features);
        }
        if (output_raw_shapes.size() == 1) {
            CHECK(output_raw_shapes.at(0) == this->m_out_features);
        }

        arma::fmat &result = output->slice(0);
        result = input_vec * weight_data_t;
        if (this->m_use_bias) {
            CHECK(!this->m_bias.empty() && this->m_bias.size() == 1)
                            << "The bias tensor is empty, but use_bias is true";

            const auto &bias_data = this->m_bias.front()->data();
            CHECK(!bias_data.empty() && bias_data.n_slices == 1 &&
                  bias_data.n_cols == this->m_out_features)
                            << "The col of bias tensor is not same to output_features_";
            const auto &bias_tensor = bias_data.slice(0);
            for (uint32_t row = 0; row < result.n_rows; ++row) {
                result.row(row) += bias_tensor;
            }
        }
    }
    return EInferStatus::EIS_InferSuccess;
}

EParseParameterAttrStatus LinearLayer::get_instance(const std::shared_ptr<RuntimeOperator> &op,
                                                    std::shared_ptr<Layer> &linear_layer) {
    CHECK(op != nullptr) << "Linear operator is nullptr";
    const auto &params = op->m_params;
    if (params.find("bias") == params.end()) {
        LOG(ERROR) << "Can not find the use bias parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingUseBias;
    }
    auto use_bias_param =
            std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
    if (use_bias_param == nullptr) {
        LOG(ERROR) << "Can not find the use bias parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingUseBias;
    }

    const auto &attr = op->m_attribute;
    CHECK(!attr.empty()) << "Operator attributes is empty";

    if (attr.find("weight") == attr.end()) {
        LOG(ERROR) << "Can not find the weight parameter";
        return EParseParameterAttrStatus::EPPAS_AttrMissingWeight;
    }

    if (use_bias_param->value) {
        if (attr.find("bias") == attr.end()) {
            LOG(ERROR) << "Can not find the bias parameter";
            return EParseParameterAttrStatus::EPPAS_AttrMissingBias;
        }
    }

    const auto &weight = attr.at("weight");
    const auto &bias = attr.at("bias");
    const auto &shapes = weight->m_shapes;
    if ((shapes.size() < 2)) {
        LOG(ERROR) << "The graph only support two dimension matrix multiply";
        return EParseParameterAttrStatus::EPPAS_AttrMissingOutFeatures;
    }

    int32_t out_features = shapes.at(0);
    int32_t in_features = shapes.at(1);
    const bool use_bias = use_bias_param->value;

    linear_layer =
            std::make_shared<LinearLayer>(in_features, out_features, use_bias);
    if (use_bias) {
        linear_layer->set_bias(bias->get_weight_data<float>());
    }

    // load weights
    linear_layer->set_weights(weight->get_weight_data<float>());
    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;
}

LayerRegistererWrapper linear_create_instance("nn.Linear", LinearLayer::get_instance);