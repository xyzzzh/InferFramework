//
// Created by xyzzzh on 2024/3/28.
//

#include "data/Tensor.hpp"

Tensor::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    m_data = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        m_raw_shapes = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        m_raw_shapes = std::vector<uint32_t>{rows, cols};
    } else {
        m_raw_shapes = std::vector<uint32_t>{rows, cols, channels};
    }
}

Tensor::Tensor(const std::vector<uint32_t> &shapes) {
    CHECK(shapes.size() == 3);
    uint32_t _channels = shapes[0];
    uint32_t _rows = shapes[1];
    uint32_t _cols = shapes[2];

    m_data = arma::fcube(_rows, _cols, _channels);

    if (_channels == 1 && _rows == 1) {
        m_raw_shapes = std::vector<uint32_t>{_cols};
    } else if (_channels == 1) {
        m_raw_shapes = std::vector<uint32_t>{_rows, _cols};
    } else {
        m_raw_shapes = std::vector<uint32_t>{_rows, _cols, _channels};
    }
}

Tensor::Tensor(const Tensor &other) {
    if (this != &other) {
        m_data = other.m_data;
        m_raw_shapes = other.m_raw_shapes;
    }
}

Tensor::Tensor(Tensor &&other) noexcept {
    if (this != &other) {
        m_data = std::move(other.m_data);
        m_raw_shapes = other.m_raw_shapes;
    }
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        m_data = std::move(other.m_data);
        m_raw_shapes = other.m_raw_shapes;
    }
    return *this;
}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        m_data = std::move(other.m_data);
        m_raw_shapes = other.m_raw_shapes;
    }
    return *this;
}

uint32_t Tensor::rows() const {
    CHECK(!m_data.empty());
    return m_data.n_rows;
}

uint32_t Tensor::cols() const {
    CHECK(!m_data.empty());
    return m_data.n_cols;
}

uint32_t Tensor::channels() const {
    CHECK(!m_data.empty());
    return m_data.n_slices;
}

uint32_t Tensor::size() const {
    CHECK(!m_data.empty());
    return m_data.size();
}

void Tensor::set_data(const arma::fcube &data) {
    CHECK(data.n_rows == m_data.n_rows) << data.n_rows << " != " << m_data.n_rows;
    CHECK(data.n_cols == m_data.n_cols) << data.n_cols << " != " << m_data.n_cols;
    CHECK(data.n_slices == m_data.n_slices) << data.n_slices << " != " << m_data.n_slices;

    m_data = data;
}

bool Tensor::empty() {
    return m_data.empty();
}

float Tensor::index(uint32_t offset) const {
    CHECK(offset < m_data.size()) << "Tensor capacity is not enough!";
    return m_data.at(offset);
}

float &Tensor::index(uint32_t offset) {
    CHECK(offset < m_data.size()) << "Tensor capacity is not enough!";
    return m_data.at(offset);
}

std::vector<uint32_t> Tensor::shapes() const {
    CHECK(!m_data.empty());
    return {channels(), rows(), cols()};
}

const std::vector<uint32_t> Tensor::raw_shapes() const {
    CHECK(!m_raw_shapes.empty());
    return m_raw_shapes;
}

arma::fcube &Tensor::data() {
    return m_data;
}

const arma::fcube Tensor::data() const {
    return m_data;
}

arma::fmat &Tensor::slice(uint32_t channel) {
    CHECK_LT(channel, channels());
    return m_data.slice(channel);
}

const arma::fmat &Tensor::slice(uint32_t channel) const {
    CHECK_LT(channel, channels());
    return m_data.slice(channel);
}

float Tensor::at(uint32_t channel, uint32_t row, uint32_t col) const {
    CHECK_LT(channel, channels());
    CHECK_LT(row, rows());
    CHECK_LT(col, cols());

    return m_data.at(row, col, channel);
}

float &Tensor::at(uint32_t channel, uint32_t row, uint32_t col) {
    CHECK_LT(channel, channels());
    CHECK_LT(row, rows());
    CHECK_LT(col, cols());

    return m_data.at(row, col, channel);
}

void Tensor::padding(const std::vector<uint32_t> &pads, float padding_value) {
    CHECK(!m_data.empty());
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
    ) = m_data;

    m_data = std::move(new_data);
}

void Tensor::fill(float value) {
    CHECK(!m_data.empty());
    m_data.fill(value);
}

void Tensor::fill(const std::vector<float> &values, bool row_major) {
    CHECK(!m_data.empty());
    const uint32_t total_elements = m_data.size();
    CHECK_EQ(values.size(), total_elements);

    if (row_major) {
        const uint32_t _rows = rows();
        const uint32_t _cols = cols();
        const uint32_t _planes = _rows * _cols;
        const uint32_t _channels = channels();

        for (uint32_t i = 0; i < _channels; i++) {
            auto &channel_data = m_data.slice(i);
            const arma::fmat &_channel_data = arma::fmat(values.data() + i * _planes, _cols, _rows);
            channel_data = _channel_data.t();   // 注意转置
        }
    } else {
        std::copy(values.begin(), values.end(), this->m_data.memptr());
    }
}

void Tensor::ones() {
    CHECK(!m_data.empty());
    m_data.fill(1.0f);
}

void Tensor::rand() {
    CHECK(!m_data.empty());
    m_data.randn();
}

void Tensor::show() {
    for (uint32_t i = 0; i < channels(); i++) {
        LOG(INFO) << "Channel: " << i;
        LOG(INFO) << "\n" << m_data.slice(i);
    }
    LOG(INFO) << "\n";
}

void Tensor::show_shapes() {
    LOG(INFO) << "Tensor shapes: ";
    for (auto num: m_raw_shapes) {
        LOG(INFO) << num << "\t";
    }
    LOG(INFO) << "\n";
}

void Tensor::reshape(const std::vector<uint32_t> &shapes, bool row_major) {
    CHECK(!this->m_data.empty());
    CHECK(!shapes.empty());
    const uint32_t origin_size = this->size();
    const uint32_t current_size =
            std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
    CHECK(shapes.size() <= 3);
    CHECK(current_size == origin_size);

    std::vector<float> values;
    if (row_major) {
        values = this->values(true);
    }
    if (shapes.size() == 3) {
        this->m_data.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
        this->m_raw_shapes = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
        this->m_data.reshape(shapes.at(0), shapes.at(1), 1);
        this->m_raw_shapes = {shapes.at(0), shapes.at(1)};
    } else {
        this->m_data.reshape(1, shapes.at(0), 1);
        this->m_raw_shapes = {shapes.at(0)};
    }

    if (row_major) {
        this->fill(values, true);
    }
}

void Tensor::flatten(bool row_major) {
    CHECK(!m_data.empty());
    const uint32_t _size = m_data.size();
    reshape({_size}, row_major);
}

void Tensor::transform(const std::function<float(float)> &filter) {
    CHECK(!m_data.empty());
    m_data.transform(filter);
}

std::shared_ptr<Tensor> Tensor::clone() {
    return std::make_shared<Tensor>(*this);
}

float *Tensor::raw_ptr() {
    CHECK(!m_data.empty());
    return m_data.memptr();
}

float *Tensor::matrix_raw_ptr(uint32_t index) {
    CHECK_LT(index, this->channels());
    uint32_t offset = index * this->rows() * this->cols();
    CHECK_LE(offset, this->size());
    float *mem_ptr = this->raw_ptr() + offset;
    return mem_ptr;
}

void Tensor::review(const std::vector<uint32_t> &shapes) {
    CHECK(!m_data.empty());
    const uint32_t target_channels = shapes[0];
    const uint32_t target_rows = shapes[1];
    const uint32_t target_cols = shapes[2];

    CHECK_EQ(m_data.size(), target_rows * target_cols * target_channels);
    arma::fcube new_data(target_rows, target_cols, target_channels);

    const uint32_t plane_size = target_rows * target_cols;
    for (uint32_t c = 0; c < channels(); c++) {
        const arma::fmat &_channel_data = m_data.slice(c);
        for (uint32_t c_ = 0; c_ < m_data.n_cols; ++c_) {
            const float *col_ptr = _channel_data.colptr(c_);
            for (uint32_t r = 0; r < m_data.n_rows; ++r) {
                const uint32_t pos_index =
                        c * m_data.n_rows * m_data.n_cols + r * m_data.n_cols + c_;
                const uint32_t _ch = pos_index / plane_size;
                const uint32_t _row = (pos_index - _ch * plane_size) / target_cols;
                const uint32_t _col = (pos_index - _ch * plane_size - _row * target_cols);
                CHECK(_ch < new_data.n_slices && _col < new_data.n_cols &&
                      _row < new_data.n_rows);
                new_data.at(_row, _col, _ch) = *(col_ptr + r);
            }
        }
    }
    m_data = std::move(new_data);
}


std::vector<float> Tensor::values(bool row_major) {
    CHECK_EQ(this->m_data.empty(), false);
    std::vector<float> values(this->m_data.size());

    if (!row_major) {
        std::copy(this->m_data.mem, this->m_data.mem + this->m_data.size(),
                  values.begin());
    } else {
        uint32_t index = 0;
        for (uint32_t c = 0; c < this->m_data.n_slices; ++c) {
            const arma::fmat &channel = this->m_data.slice(c).t();
            std::copy(channel.begin(), channel.end(), values.begin() + index);
            index += channel.size();
        }
        CHECK_EQ(index, values.size());
    }
    return values;
}
