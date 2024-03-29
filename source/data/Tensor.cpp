//
// Created by xyzzzh on 2024/3/28.
//

#include "data/Tensor.hpp"

Tensor::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    data_ = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        raw_shapes_ = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        raw_shapes_ = std::vector<uint32_t>{rows, cols};
    } else {
        raw_shapes_ = std::vector<uint32_t>{rows, cols, channels};
    }
    std::cout << channels << "\t" << rows << "\t" << cols << std::endl;
}

Tensor::Tensor(const std::vector<uint32_t> &shapes) {
    CHECK(shapes.size() == 3);
    uint32_t _channels = shapes[0];
    uint32_t _rows = shapes[1];
    uint32_t _cols = shapes[2];

    data_ = arma::fcube(_rows, _cols, _channels);

    if (_channels == 1 && _rows == 1) {
        raw_shapes_ = std::vector<uint32_t>{_cols};
    } else if (_channels == 1) {
        raw_shapes_ = std::vector<uint32_t>{_rows, _cols};
    } else {
        raw_shapes_ = std::vector<uint32_t>{_rows, _cols, _channels};
    }
}

Tensor::Tensor(const Tensor &other) {
    if (this != &other) {
        data_ = other.data_;
        raw_shapes_ = other.raw_shapes_;
    }
}

Tensor::Tensor(Tensor &&other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        raw_shapes_ = other.raw_shapes_;
    }
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        raw_shapes_ = other.raw_shapes_;
    }
    return *this;
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        data_ = std::move(other.data_);
        raw_shapes_ = other.raw_shapes_;
    }
    return *this;
}

uint32_t Tensor::rows() const {
    CHECK(!data_.empty());
    return data_.n_rows;
}

uint32_t Tensor::cols() const {
    CHECK(!data_.empty());
    return data_.n_cols;
}

uint32_t Tensor::channels() const {
    CHECK(!data_.empty());
    return data_.n_slices;
}

uint32_t Tensor::size() const {
    CHECK(!data_.empty());
    return data_.size();
}

void Tensor::set_data(const arma::fcube &data) {
    CHECK(data.n_rows == data_.n_rows) << data.n_rows << " != " << data_.n_rows;
    CHECK(data.n_cols == data_.n_cols) << data.n_cols << " != " << data_.n_cols;
    CHECK(data.n_slices == data_.n_slices) << data.n_slices << " != " << data_.n_slices;

    data_ = data;
}

bool Tensor::empty() {
    return data_.empty();
}

float Tensor::index(uint32_t offset) const {
    CHECK(offset < data_.size()) << "Tensor capacity is not enough!";
    return data_.at(offset);
}

float &Tensor::index(uint32_t offset) {
    CHECK(offset < data_.size()) << "Tensor capacity is not enough!";
    return data_.at(offset);
}

std::vector<uint32_t> Tensor::shapes() const {
    CHECK(!data_.empty());
    return {channels(), rows(), cols()};
}

const std::vector<uint32_t> Tensor::raw_shapes() const {
    CHECK(!raw_shapes_.empty());
    return raw_shapes_;
}

arma::fcube &Tensor::data() {
    return data_;
}

const arma::fcube Tensor::data() const {
    return data_;
}

arma::fmat &Tensor::slice(uint32_t channel) {
    CHECK_LT(channel, channels());
    return data_.slice(channel);
}

const arma::fmat &Tensor::slice(uint32_t channel) const {
    CHECK_LT(channel, channels());
    return data_.slice(channel);
}

float Tensor::at(uint32_t channel, uint32_t row, uint32_t col) const {
    CHECK_LT(channel, channels());
    CHECK_LT(row, rows());
    CHECK_LT(col, cols());

    return data_.at(row, col, channel);
}

float &Tensor::at(uint32_t channel, uint32_t row, uint32_t col) {
    CHECK_LT(channel, channels());
    CHECK_LT(row, rows());
    CHECK_LT(col, cols());

    return data_.at(row, col, channel);
}

void Tensor::padding(const std::vector<uint32_t> &pads, float padding_value) {
    CHECK(!data_.empty());
    CHECK_EQ(pads.size(), 4);
    uint32_t pad_rows1 = pads[0];
    uint32_t pad_rows2 = pads[1];
    uint32_t pad_cols1 = pads[2];
    uint32_t pad_cols2 = pads[3];

    arma::fcube new_data(rows() + pad_rows1 + pad_rows2, cols() + pad_cols1 + pad_cols2, channels());

    new_data.fill(padding_value);
    new_data.subcube(
            pad_rows1, pad_cols1, 0,
            new_data.n_rows - pad_rows2 - 1, new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1
    ) = data_;

    data_ = std::move(new_data);
}

void Tensor::fill(float value) {
    CHECK(!data_.empty());
    data_.fill(value);
}

void Tensor::fill(const std::vector<float> &values) {
    CHECK(!data_.empty());
    const uint32_t total_elements = data_.size();
    CHECK_EQ(values.size(), total_elements);

    const uint32_t _rows = rows();
    const uint32_t _cols = cols();
    const uint32_t _planes = _rows * _cols;
    const uint32_t _channels = channels();

    for (uint32_t i = 0; i < _channels; i++) {
        auto &channel_data = data_.slice(i);
        const arma::fmat &_channel_data = arma::fmat(values.data() + i * _planes, _cols, _rows);
        channel_data = _channel_data.t();   // 注意转置
    }
}

void Tensor::ones() {
    CHECK(!data_.empty());
    data_.fill(1.0f);
}

void Tensor::rand() {
    CHECK(!data_.empty());
    data_.randn();
}

void Tensor::show() {
    for (uint32_t i = 0; i < channels(); i++) {
        LOG(INFO) << "Channel: " << i;
        LOG(INFO) << "\n" << data_.slice(i);
    }
    LOG(INFO) << "\n";
}

void Tensor::show_shapes() {
    LOG(INFO) << "Tensor shapes: ";
    for(auto num : raw_shapes_){
        LOG(INFO) << num << "\t";
    }
    LOG(INFO) << "\n";
}

void Tensor::reshape(const std::vector<uint32_t> &shapes, bool row_major) {
    CHECK(!data_.empty());
    CHECK(!shapes.empty());
    CHECK_LE(shapes.size(), 3);
    const uint32_t origin_size = size();

    uint32_t new_size = 1;
    for (auto &s: shapes) {
        new_size *= s;
    }
    CHECK_EQ(origin_size, new_size);

    if (row_major) {
        // channel row col
        std::vector<uint32_t> target_shapes;
        if (shapes.size() == 3) {
            target_shapes = {shapes[0], shapes[1], shapes[2]};
            raw_shapes_ = {shapes[0], shapes[1], shapes[2]};
        } else if (shapes.size() == 2) {
            target_shapes = {1, shapes[0], shapes[1]};
            raw_shapes_ = {shapes[0], shapes[1]};
        } else {
            target_shapes = {1, shapes[0], 1};
            raw_shapes_ = {shapes[0]};
        }
        this->review(target_shapes);
    } else {
        if (shapes.size() == 3) {
            data_.reshape(shapes[1], shapes[2], shapes[0]);
            raw_shapes_ = {shapes[0], shapes[1], shapes[2]};
        } else if (shapes.size() == 2) {
            data_.reshape(shapes[0], shapes[1], 1);
            raw_shapes_ = {shapes[0], shapes[1]};
        } else {
            data_.reshape(shapes[0], 1, 1);
            raw_shapes_ = {shapes[0]};
        }
    }
}

void Tensor::flatten(bool row_major) {
    CHECK(!data_.empty());
    const uint32_t _size = data_.size();
    reshape({_size}, row_major);
}

void Tensor::transform(const std::function<float(float)> &filter) {
    CHECK(!data_.empty());
    data_.transform(filter);
}

std::shared_ptr<Tensor> Tensor::clone() {
    return std::make_shared<Tensor>(*this);
}

const float *Tensor::raw_ptr() const {
    CHECK(!data_.empty());
    return data_.memptr();
}

void Tensor::review(const std::vector<uint32_t> &shapes) {
    CHECK(!data_.empty());
    const uint32_t target_channels = shapes[0];
    const uint32_t target_rows = shapes[1];
    const uint32_t target_cols = shapes[2];

    CHECK_EQ(data_.size(), target_rows * target_cols * target_channels);
    arma::fcube new_data(target_rows, target_cols, target_channels);

    const uint32_t plane_size = target_rows * target_cols;
    for (uint32_t c = 0; c < channels(); c++) {
        const arma::fmat &_channel_data = data_.slice(c);
        for (uint32_t c_ = 0; c_ < data_.n_cols; ++c_) {
            const float *col_ptr = _channel_data.colptr(c_);
            for (uint32_t r = 0; r < data_.n_rows; ++r) {
                const uint32_t pos_index =
                        c * data_.n_rows * data_.n_cols + r * data_.n_cols + c_;
                const uint32_t _ch = pos_index / plane_size;
                const uint32_t _row = (pos_index - _ch * plane_size) / target_cols;
                const uint32_t _col = (pos_index - _ch * plane_size - _row * target_cols);
                CHECK(_ch < new_data.n_slices && _col < new_data.n_cols &&
                      _row < new_data.n_rows);
                new_data.at(_row, _col, _ch) = *(col_ptr + r);
            }
        }
    }
    data_ = std::move(new_data);
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

std::shared_ptr<Tensor> tensor_create(uint32_t channels, uint32_t rows, uint32_t cols) {
    return std::make_shared<Tensor>(channels, rows, cols);
}

std::shared_ptr<Tensor> tensor_create(const std::vector<uint32_t> &shapes) {
    CHECK_EQ(shapes.size(), 3);
    return std::make_shared<Tensor>(shapes[0], shapes[1], shapes[2]);
}
