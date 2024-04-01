//
// Created by xyzzzh on 2024/4/1.
//

#ifndef INFERFRAMEWORK_UTILS_HPP
#define INFERFRAMEWORK_UTILS_HPP

#include "Common.hpp"
#include "runtime/RuntimeOperator.hpp"

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
tensor_broadcast(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2);

std::shared_ptr<Tensor>
tensor_padding(const std::shared_ptr<Tensor> &tensor, const std::vector<uint32_t> &pads, float padding_value);

bool tensor_is_same(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2);

std::shared_ptr<Tensor> tensor_add(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2);

void tensor_add(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2,
                const std::shared_ptr<Tensor> &output_tensor);

std::shared_ptr<Tensor> tensor_multiply(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2);

void tensor_multiply(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2,
                     const std::shared_ptr<Tensor> &output_tensor);

std::shared_ptr<Tensor> tensor_create(uint32_t channels, uint32_t rows, uint32_t cols);

std::shared_ptr<Tensor> tensor_create(const std::vector<uint32_t> &shapes);

std::pair<size_t, size_t> get_mat_size(std::ifstream &file, char split_char);

arma::fmat load_data(const std::string &file_path, char split_char = ',');

std::string shape_str(const std::vector<uint32_t> &shapes);

std::string shape_str(const std::vector<int> &shapes);

/// 如果图是第一次运行，则根据节点输入operand的形状准备好后续Layer计算中所需要的Tensor
/// 如果图是第二次及以后运行，则检查输入operand的形状和operand中张量的形状是否匹配
void init_operator_input(const std::vector<std::shared_ptr<RuntimeOperator>> &operators);

/// 如果图是第一次运行，则根据节点输出operand的形状准备好后续Layer计算中所需要的Tensor
/// 如果图是第二次及以后运行，则检查输出operand的形状和operand中张量的形状是否匹配

void init_operator_output(const std::vector<pnnx::Operator *> &pnnx_operators,
                        const std::vector<std::shared_ptr<RuntimeOperator>> &operators);

#endif //INFERFRAMEWORK_UTILS_HPP
