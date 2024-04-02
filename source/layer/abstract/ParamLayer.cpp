//
// Created by xyzzzh on 2024/4/2.
//

#include "layer/abstract/ParamLayer.hpp"

void
ParamLayer::init_weight_param(const uint32_t param_count, const uint32_t param_channel,
                              const uint32_t param_height, const uint32_t param_width) {
    this->m_weights = std::vector<std::shared_ptr<Tensor>>(param_count);
    for (uint32_t i = 0; i < param_count; i++) {
        this->m_weights[i] = std::make_shared<Tensor>(param_channel, param_height, param_width);
    }
}

void ParamLayer::init_bias_param(const uint32_t param_count, const uint32_t param_channel,
                                 const uint32_t param_height, const uint32_t param_width) {
    this->m_bias = std::vector<std::shared_ptr<Tensor>>(param_count);
    for (uint32_t i = 0; i < param_count; i++) {
        this->m_bias[i] = std::make_shared<Tensor>(param_channel, param_height, param_width);
    }
}

const std::vector<std::shared_ptr<Tensor>> &ParamLayer::weights() const {
    return this->m_weights;
}

const std::vector<std::shared_ptr<Tensor>> &ParamLayer::bias() const {
    return this->m_bias;
}

void ParamLayer::set_weights(const std::vector<std::shared_ptr<Tensor>> &weights) {
    CHECK(weights.size() == this->m_weights.size());
    for (uint32_t i = 0; i < weights.size(); ++i) {
        CHECK(this->m_weights[i] != nullptr);
        CHECK(this->m_weights[i]->rows() == weights[i]->rows());
        CHECK(this->m_weights[i]->cols() == weights[i]->cols());
        CHECK(this->m_weights[i]->channels() == weights[i]->channels());
    }
    this->m_weights = weights;
}

void ParamLayer::set_weights(const std::vector<float> &weights) {
    const uint32_t elem_size = weights.size();

    uint32_t weight_size = 0;
    const uint32_t batch_size = this->m_weights.size();
    for (uint32_t i = 0; i < batch_size; i++) {
        weight_size += this->m_weights[i]->size();
    }

    CHECK(weight_size == elem_size);
    CHECK(elem_size % batch_size == 0);
    const uint32_t blob_size = elem_size / batch_size;
    for (uint32_t i = 0; i < batch_size; i++) {
        const uint32_t start_offset = i * blob_size;
        const uint32_t end_offset = start_offset + blob_size;
        const auto &sub_value = std::vector<float>{weights.begin() + start_offset, weights.begin() + end_offset};
        this->m_weights[i]->fill(sub_value);
    }
}

void ParamLayer::set_bias(const std::vector<std::shared_ptr<Tensor>> &bias) {
    CHECK(bias.size() == this->m_bias.size());
    for (uint32_t i = 0; i < bias.size(); ++i) {
        CHECK(this->m_bias[i] != nullptr);
        CHECK(this->m_bias[i]->rows() == bias[i]->rows());
        CHECK(this->m_bias[i]->cols() == bias[i]->cols());
        CHECK(this->m_bias[i]->channels() == bias[i]->channels());
    }
    this->m_bias = bias;
}

void ParamLayer::set_bias(const std::vector<float> &bias) {
    const uint32_t elem_size = bias.size();

    uint32_t weight_size = 0;
    const uint32_t batch_size = this->m_bias.size();
    for (uint32_t i = 0; i < batch_size; i++) {
        weight_size += this->m_bias[i]->size();
    }

    CHECK(weight_size == elem_size);
    CHECK(elem_size % batch_size == 0);
    const uint32_t blob_size = elem_size / batch_size;
    for (uint32_t i = 0; i < batch_size; i++) {
        const uint32_t start_offset = i * blob_size;
        const uint32_t end_offset = start_offset + blob_size;
        const auto &sub_value = std::vector<float>{bias.begin() + start_offset, bias.begin() + end_offset};
        this->m_bias[i]->fill(sub_value);
    }
}
