//
// Created by xyzzzh on 2024/4/1.
//
#include "Utils.hpp"



std::shared_ptr<Tensor> tensor_create(uint32_t channels, uint32_t rows, uint32_t cols) {
    return std::make_shared<Tensor>(channels, rows, cols);
}

std::shared_ptr<Tensor> tensor_create(const std::vector<uint32_t> &shapes) {
    CHECK_EQ(shapes.size(), 3);
    return std::make_shared<Tensor>(shapes[0], shapes[1], shapes[2]);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
tensor_broadcast(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        return {tensor1, tensor2};
    } else {
        CHECK(tensor1->channels() == tensor2->channels());
        if (tensor2->rows() == 1 && tensor2->cols() == 1) {
            std::shared_ptr<Tensor> new_tensor =
                    tensor_create(tensor2->channels(), tensor1->rows(), tensor1->cols());
            CHECK(tensor2->size() == tensor2->channels());
            for (uint32_t c = 0; c < tensor2->channels(); ++c) {
                new_tensor->slice(c).fill(tensor2->index(c));
            }
            return {tensor1, new_tensor};
        } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
            std::shared_ptr<Tensor> new_tensor =
                    tensor_create(tensor1->channels(), tensor2->rows(), tensor2->cols());
            CHECK(tensor1->size() == tensor1->channels());
            for (uint32_t c = 0; c < tensor1->channels(); ++c) {
                new_tensor->slice(c).fill(tensor1->index(c));
            }
            return {new_tensor, tensor2};
        } else {
            LOG(FATAL) << "Broadcast shape is not adapting!";
            return {tensor1, tensor2};
        }
    }
}

std::shared_ptr<Tensor>
tensor_padding(const std::shared_ptr<Tensor> &tensor, const std::vector<uint32_t> &pads, float padding_value) {
    CHECK(tensor != nullptr && !tensor->empty());
    CHECK(pads.size() == 4);
    uint32_t pad_rows1 = pads[0];  // up
    uint32_t pad_rows2 = pads[1];  // bottom
    uint32_t pad_cols1 = pads[2];  // left
    uint32_t pad_cols2 = pads[3];  // right

    std::shared_ptr<Tensor> output_tensor = std::make_shared<Tensor>(
            tensor->channels(), tensor->rows() + pad_rows1 + pad_rows2, tensor->cols() + pad_cols1 + pad_cols2);
    output_tensor->fill(padding_value);

    const uint32_t _channels = tensor->channels();
    for (uint32_t _channel = 0; _channel < _channels; _channel++) {
        const arma::fmat &in_channel = tensor->slice(_channel);
        arma::fmat &output_channel = output_tensor->slice(_channel);
        const uint32_t in_channel_width = in_channel.n_cols;
        const uint32_t in_channel_height = in_channel.n_rows;

        for (uint32_t w = 0; w < in_channel_width; ++w) {
            float *output_channel_ptr =
                    const_cast<float *>(output_channel.colptr(w + pad_cols1));
            const float *in_channel_ptr = in_channel.colptr(w);
            for (uint32_t h = 0; h < in_channel_height; ++h) {
                const float value = *(in_channel_ptr + h);
                *(output_channel_ptr + h + pad_rows1) = value;
            }
        }
    }
    return output_tensor;
}

bool tensor_is_same(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() != tensor2->shapes()) {
        return false;
    }
    bool is_same = arma::approx_equal(tensor1->data(), tensor2->data(), "absdiff", 1e-5);
    return is_same;
}

std::shared_ptr<Tensor>
tensor_add(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        std::shared_ptr<Tensor> output_tensor = tensor_create(tensor1->shapes());
        output_tensor->set_data(tensor1->data() + tensor2->data());
        return output_tensor;
    } else {
        // broadcast
        CHECK(tensor1->channels() == tensor2->channels())
                        << "Tensors shape are not adapting";
        const auto &[input_tensor1, input_tensor2] =
                tensor_broadcast(tensor1, tensor2);
        CHECK(input_tensor1->shapes() == input_tensor2->shapes());
        std::shared_ptr<Tensor> output_tensor = tensor_create(input_tensor1->shapes());
        output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
        return output_tensor;
    }
}

void tensor_add(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2,
                const std::shared_ptr<Tensor> &output_tensor) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        output_tensor->set_data(tensor1->data() + tensor2->data());
    } else {
        // broadcast
        CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
        const auto &[input_tensor1, input_tensor2] = tensor_broadcast(tensor1, tensor2);
        CHECK(input_tensor1->shapes() == input_tensor2->shapes() && output_tensor->shapes() == input_tensor1->shapes());
        output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
    }
}

std::shared_ptr<Tensor>
tensor_multiply(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        std::shared_ptr<Tensor> output_tensor = tensor_create(tensor1->shapes());
        output_tensor->set_data(tensor1->data() % tensor2->data());
        return output_tensor;
    } else {
        // broadcast
        CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
        const auto &[input_tensor1, input_tensor2] = tensor_broadcast(tensor1, tensor2);
        CHECK(input_tensor1->shapes() == input_tensor2->shapes());
        std::shared_ptr<Tensor> output_tensor = tensor_create(input_tensor1->shapes());
        output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
        return output_tensor;
    }
}

void tensor_multiply(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2,
                     const std::shared_ptr<Tensor> &output_tensor) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
    if (tensor1->shapes() == tensor2->shapes()) {
        output_tensor->set_data(tensor1->data() % tensor2->data());
    } else {
        // broadcast
        CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
        const auto &[input_tensor1, input_tensor2] = tensor_broadcast(tensor1, tensor2);
        CHECK(input_tensor1->shapes() == input_tensor2->shapes() && output_tensor->shapes() == input_tensor1->shapes());
        output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
    }
}
