//
// Created by xyzzzh on 2024/4/2.
//

#ifndef INFERFRAMEWORK_LAYER_HPP
#define INFERFRAMEWORK_LAYER_HPP

#include "Common.hpp"
#include "runtime/RuntimeOperator.hpp"

class Layer {
public:
    explicit Layer(std::string layer_name) : m_layer_name(std::move(layer_name)) {}

    virtual ~Layer() = default;

    // Layer的执行函数
    virtual EInferStatus Forward(
            const std::vector<std::shared_ptr<Tensor>> &inputs,
            std::vector<std::shared_ptr<Tensor>> &outputs
    );

    // Layer的执行函数
    virtual EInferStatus Forward();

    // 返回层的权重
    virtual const std::vector<std::shared_ptr<Tensor>> &weights() const;

    // 返回层的偏移量
    virtual const std::vector<std::shared_ptr<Tensor>> &bias() const;

    // 设置Layer的权重
    virtual void set_weights(const std::vector<std::shared_ptr<Tensor>> &weights);

    virtual void set_weights(const std::vector<float> &weights);

    // 设置Layer的偏移量
    virtual void set_bias(const std::vector<std::shared_ptr<Tensor>> &bias);

    virtual void set_bias(const std::vector<float> &bias);

    // 返回层的名称
    virtual const std::string layer_name() const { return this->m_layer_name; }

    // 设置层的执行算子
    void set_runtime_operator(
            const std::shared_ptr<RuntimeOperator> &runtime_operator
    ) { this->m_runtime_operator = runtime_operator; }

private:
    std::weak_ptr<RuntimeOperator> m_runtime_operator;
    std::string m_layer_name;
};


#endif //INFERFRAMEWORK_LAYER_HPP
