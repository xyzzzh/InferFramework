//
// Created by xyzzzh on 2024/4/2.
//

#ifndef INFERFRAMEWORK_PARAMLAYER_HPP
#define INFERFRAMEWORK_PARAMLAYER_HPP

#include "Common.hpp"
#include "layer/abstract/Layer.hpp"

class ParamLayer : public Layer{
public:
    explicit ParamLayer(const std::string &layer_name) : Layer(layer_name){}

    // 初始化权重空间
    void init_weight_param(const uint32_t param_count, const uint32_t param_channel,
                           const uint32_t param_height, const uint32_t param_width);

    // 初始化偏移参数
    void init_bias_param(const uint32_t param_count, const uint32_t param_channel,
                         const uint32_t param_height, const uint32_t param_width);

    // 返回层的权重
    const std::vector<std::shared_ptr<Tensor>> &weights() const override;

    // 返回层的偏移量
    const std::vector<std::shared_ptr<Tensor>> &bias() const override;

    // 设置Layer的权重
    void set_weights(const std::vector<std::shared_ptr<Tensor>> &weights) override;

    void set_weights(const std::vector<float> &weights) override;

    // 设置Layer的偏移量
    void set_bias(const std::vector<std::shared_ptr<Tensor>> &bias) override;

    void set_bias(const std::vector<float> &bias) override;

private:
    std::vector<std::shared_ptr<Tensor>> m_weights;
    std::vector<std::shared_ptr<Tensor>> m_bias;
};


#endif //INFERFRAMEWORK_PARAMLAYER_HPP
