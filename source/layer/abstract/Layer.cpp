//
// Created by xyzzzh on 2024/4/2.
//

#include "layer/abstract/Layer.hpp"

EInferStatus
Layer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs) {
    return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
}

EInferStatus Layer::forward() {

    LOG_IF(FATAL, this->m_runtime_operator.expired()) << "Runtime operator is expired or nullptr";

    // 准备节点layer计算所需要的输入
    const auto &runtime_operator = this->m_runtime_operator.lock();
    const std::vector<std::shared_ptr<RuntimeOperand>> &input_operand_datas = runtime_operator->m_input_operands_seq;

    // layer的输入
    std::vector<std::shared_ptr<Tensor>> layer_input_datas;
    for (const auto &input_operand_data: input_operand_datas) {
        for (const auto &input_data: input_operand_data->m_data) {
            layer_input_datas.push_back(input_data);
        }
    }

    const std::shared_ptr<RuntimeOperand> &output_operand_datas = runtime_operator->m_output_operands;

    CHECK(!layer_input_datas.empty()) << runtime_operator->m_name << " Layer input data is empty";
    CHECK(output_operand_datas != nullptr && !output_operand_datas->m_data.empty()) << "Layer output data is empty";

    // 执行operator当中的layer计算过程
    // layer的计算结果存放在current_op->output_operands->m_data中
    EInferStatus status = runtime_operator->m_layer->forward(layer_input_datas, output_operand_datas->m_data);

    return status;
}

const std::vector<std::shared_ptr<Tensor>> &Layer::weights() const {
    return {};
}

const std::vector<std::shared_ptr<Tensor>> &Layer::bias() const {
    return {};
}

void Layer::set_weights(const std::vector<std::shared_ptr<Tensor>> &weights) {}

void Layer::set_weights(const std::vector<float> &weights) {}

void Layer::set_bias(const std::vector<std::shared_ptr<Tensor>> &bias) {}

void Layer::set_bias(const std::vector<float> &bias) {}
