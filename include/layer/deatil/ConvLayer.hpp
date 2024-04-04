//
// Created by xyzzzh on 2024/4/3.
//

#ifndef INFERFRAMEWORK_CONVLAYER_HPP
#define INFERFRAMEWORK_CONVLAYER_HPP

#include "Common.hpp"
#include "runtime/RuntimeGraph.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/abstract/ParamLayer.hpp"

class ConvLayer : public ParamLayer {
public:
    explicit ConvLayer(uint32_t output_channel, uint32_t in_channel,
                       uint32_t kernel_h, uint32_t kernel_w,
                       uint32_t padding_h, uint32_t padding_w,
                       uint32_t stride_h, uint32_t stride_w,
                       uint32_t groups, bool use_bias = true);

    static EParseParameterAttrStatus get_instance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &conv_layer);

    EInferStatus forward(
            const std::vector<std::shared_ptr<Tensor>> &inputs,
            std::vector<std::shared_ptr<Tensor>> &outputs) override;

    // 初始化kernel的IM2COL排布
    void init_IM2COL_weight();

private:
    void conv_GEMM_bias(const arma::fmat &input_matrix, std::shared_ptr<Tensor> output_tensor,
                      uint32_t group, uint32_t kernel_index,
                      uint32_t kernel_count_group, const arma::frowvec &kernel,
                      uint32_t output_w, uint32_t output_h) const;

    arma::fmat IM2COL(std::shared_ptr<Tensor> input, uint32_t kernel_w, uint32_t kernel_h,
                      uint32_t input_w, uint32_t input_h, uint32_t input_c_group,
                      uint32_t group, uint32_t row_len, uint32_t col_len) const;

private:
    bool m_use_bias = false;
    uint32_t m_groups = 1;
    uint32_t m_padding_h = 0;
    uint32_t m_padding_w = 0;
    uint32_t m_stride_h = 1;
    uint32_t m_stride_w = 1;
    std::vector<arma::frowvec> m_kernel_matrix_arr;
};


#endif //INFERFRAMEWORK_CONVLAYER_HPP
