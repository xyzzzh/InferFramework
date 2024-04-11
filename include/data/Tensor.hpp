//
// Created by xyzzzh on 2024/3/28.
//

#ifndef INFERFRAMEWORK_TENSOR_HPP
#define INFERFRAMEWORK_TENSOR_HPP

#include <armadillo>
#include <vector>
#include <memory>
#include <glog/logging.h>

// 默认为float
class Tensor {
public:
    // channels     张量的通道数     uint32_t
    // rows         张量的行数       uint32_t
    // cols         张量的列数       uint32_t

    // 构造函数
    explicit Tensor() = default;

    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    explicit Tensor(const std::vector<uint32_t> &shapes);

    // 拷贝构造
    Tensor(const Tensor &other);

    // 移动构造
    Tensor(Tensor &&other) noexcept;

    // 移动赋值
    Tensor &operator=(Tensor &&other) noexcept;

    // 拷贝赋值
    Tensor &operator=(const Tensor &other);

    // 获取rows
    uint32_t rows() const;

    // 获取cols
    uint32_t cols() const;

    // 获取channels
    uint32_t channels() const;

    // 获取m_data中元素总数
    uint32_t size() const;

    // 设置m_data
    void set_data(const arma::fcube &data);

    // 获取m_data是否为空
    bool empty();

    // 返回张量中offset位置的元素
    float index(uint32_t offset) const;

    float &index(uint32_t offset);

    // 获取形状{channels(), rows(), cols()}
    std::vector<uint32_t> shapes() const;

    // 获取m_raw_shape
    const std::vector<uint32_t> raw_shapes() const;

    // 获取m_data
    arma::fcube &data();

    const arma::fcube data() const;

    // 返回张量第channel通道中的数据
    arma::fmat &slice(uint32_t channel);

    // 获取m_data中第channel通道的fmat数据
    const arma::fmat &slice(uint32_t channel) const;

    // 获取m_data[channel, row, col]
    float at(uint32_t channel, uint32_t row, uint32_t col) const;

    float &at(uint32_t channel, uint32_t row, uint32_t col);

    // 在m_data四周填充padding_value
    // pads中顺序依次为{上, 下, 左, 右}
    void padding(const std::vector<uint32_t> &pads, float padding_value);

    // 用value填充m_data
    void fill(float value);

    // 用给定的浮点数数组values填充m_data。
    void fill(const std::vector<float> &values, bool row_major = true);

    // 以常量1初始化张量
    void ones();

    // 以随机值初始化张量
    void rand();

    // 打印张量
    void show();

    // 打印张量shapes
    void show_shapes();

    // 张量的实际尺寸大小的reshape
    void reshape(const std::vector<uint32_t> &shapes, bool row_major = false);

    // 对m_data进行压平
    void flatten(bool row_major = false);

    // 对张量中的元素进行过滤
    void transform(const std::function<float(float)> &filter);

    // 返回一个深拷贝后的张量
    std::shared_ptr<Tensor> clone();

    // 返回数据的原始指针
    float *raw_ptr();

    // 返回第index个矩阵的起始地址
    float *matrix_raw_ptr(uint32_t index);

    // 返回Tensor内的所有数据
    std::vector<float> values(bool row_major = true);

private:
    // 根据给定的形状参数重新调整Tensor对象中的数据。
    // shapes: 一个包含目标形状的数组，预期为[target_channels, target_rows, target_cols]。
    void review(const std::vector<uint32_t> &shapes);

    // 张量数据的实际尺寸大小
    // {rows, cols, channels}
    std::vector<uint32_t> m_raw_shapes;

    // 张量数据
    arma::fcube m_data;
};

#endif //INFERFRAMEWORK_TENSOR_HPP
