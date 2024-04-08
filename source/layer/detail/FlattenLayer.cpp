//
// Created by xyzzzh on 2024/4/7.
//

#include "layer/deatil/FlattenLayer.hpp"

FlattenLayer::FlattenLayer(int start_dim, int end_dim) :
        NonParamLayer("Flatten"),
        m_start_dim(start_dim),
        m_end_dim(end_dim) {}

EInferStatus FlattenLayer::forward(
        const std::vector<std::shared_ptr<Tensor>> &inputs,
        std::vector<std::shared_ptr<Tensor>> &outputs) {

    // 检查input和output的size是否合法
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the softmax layer is empty";
        return EInferStatus::EIS_InferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of the softmax layer "
                      "do not match";
        return EInferStatus::EIS_InferFailedInputOutSizeMatchError;
    }

    int total_dim = 4;  // NCHW
    int start_dim = this->m_start_dim < 0 ? total_dim + this->m_start_dim : this->m_start_dim;
    int end_dim = this->m_end_dim < 0 ? total_dim + this->m_end_dim : this->m_end_dim;


    CHECK(end_dim > start_dim) << "The end dim must greater than start dim";
    CHECK(end_dim <= 3 && start_dim >= 1) << "The end dim must less than two and start dim must greater than zero";

    const uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; i++) {
        const std::shared_ptr<Tensor> &input = inputs[i];
        if (input == nullptr || input->empty()) {
            LOG(ERROR) << "The input tensor array in the flatten layer has"
                          " an empty tensor "
                       << i << " th";
            return EInferStatus::EIS_InferFailedInputEmpty;
        }

        auto shapes = input->shapes();
        shapes.insert(shapes.begin(), batch_size);
        uint32_t elements_size = std::accumulate(shapes.begin() + start_dim, shapes.begin() + end_dim + 1, 1,
                                                 std::multiplies());

        std::shared_ptr<Tensor> output = outputs[i];
        output = tensor_clone(input);

        CHECK(input->size() == output->size())
                        << "The output and input shapes of the flatten layer do not match " << i << " th";
        outputs[i] = output;

        if (start_dim == 1 && end_dim == 3) {
            output->reshape({elements_size}, true);
        } else if (start_dim == 2 && end_dim == 3) {
            uint32_t channels = input->channels();
            output->reshape({channels, elements_size}, true);
        } else if (start_dim == 1 && end_dim == 2) {
            uint32_t cols = input->cols();
            output->reshape({elements_size, cols}, true);
        } else {
            LOG(FATAL) << "Wrong flatten dim: " << "start dim: " << start_dim << " end dim: " << end_dim;
        }
    }
    return EInferStatus::EIS_InferSuccess;
}

EParseParameterAttrStatus FlattenLayer::create_instance(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &flatten_layer) {
    CHECK(op != nullptr) << "Flatten operator is nullptr";
    const auto &params = op->m_params;

    if (params.find("end_dim") == params.end()) {
        LOG(ERROR) << "Can not find the dimension parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingDim;
    }

    if (params.find("start_dim") == params.end()) {
        LOG(ERROR) << "Can not find the dimension parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingDim;
    }

    auto start_dim =
            std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("start_dim"));

    auto end_dim =
            std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("end_dim"));

    if (start_dim == nullptr || end_dim == nullptr) {
        return EParseParameterAttrStatus::EPPAS_ParameterMissingDim;
    }

    flatten_layer =
            std::make_shared<FlattenLayer>(start_dim->value, end_dim->value);
    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;
}

LayerRegistererWrapper flattten_linear_create_instance("torch.flatten", FlattenLayer::create_instance);