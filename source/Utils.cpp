//
// Created by xyzzzh on 2024/4/1.
//
#include "Utils.hpp"


std::shared_ptr<Tensor> tensor_create(uint32_t channels, uint32_t rows, uint32_t cols) {
    return std::make_shared<Tensor>(channels, rows, cols);
}

std::shared_ptr<Tensor> tensor_create(uint32_t rows, uint32_t cols) {
    return std::make_shared<Tensor>(1, rows, cols);
}

std::shared_ptr<Tensor> tensor_create(uint32_t size) {
    return std::make_shared<Tensor>(1, 1, size);
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

std::pair<size_t, size_t> get_mat_size(std::ifstream &file, char split_char) {
    bool load_ok = file.good();
    file.clear();
    size_t fn_rows = 0;
    size_t fn_cols = 0;
    const std::ifstream::pos_type start_pos = file.tellg();

    std::string token;
    std::string line_str;
    std::stringstream line_stream;

    while (file.good() && load_ok) {
        std::getline(file, line_str);
        if (line_str.empty()) {
            break;
        }

        line_stream.clear();
        line_stream.str(line_str);
        size_t line_cols = 0;

        std::string row_token;
        while (line_stream.good()) {
            std::getline(line_stream, row_token, split_char);
            ++line_cols;
        }
        if (line_cols > fn_cols) {
            fn_cols = line_cols;
        }

        ++fn_rows;
    }
    file.clear();
    file.seekg(start_pos);
    return {fn_rows, fn_cols};
}

arma::fmat load_data(const std::string &file_path, char split_char) {
    arma::fmat _data;
    if (file_path.empty()) {
        LOG(ERROR) << "CSV file path is empty: " << file_path;
        return _data;
    }

    std::ifstream in(file_path);
    if (!in.is_open() || !in.good()) {
        LOG(ERROR) << "Open file failed: " << file_path;
        return _data;
    }

    std::string line_str;
    std::stringstream line_stream;


    const auto &[rows, cols] = get_mat_size(in, split_char);
    _data.zeros(rows, cols);

    size_t row = 0;
    while (in.good()) {
        std::getline(in, line_str);
        if (line_str.empty()) {
            break;
        }

        std::string token;
        line_stream.clear();
        line_stream.str(line_str);

        size_t col = 0;
        while (line_stream.good()) {
            std::getline(line_stream, token, split_char);
            try {
                _data.at(row, col) = std::stof(token);
            }
            catch (std::exception &e) {
                DLOG(ERROR) << "Parse CSV File meet error: " << e.what() << " row:" << row << " col:" << col;
            }
            col += 1;
            CHECK(col <= cols) << "There are excessive elements on the column";
        }

        row += 1;
        CHECK(row <= rows) << "There are excessive elements on the row";
    }
    return _data;
}

std::string shape_str(const std::vector<uint32_t> &shapes) {
    std::ostringstream ss;
    for (int i = 0; i < shapes.size(); ++i) {
        ss << shapes.at(i);
        if (i != shapes.size() - 1) {
            ss << " x ";
        }
    }
    return ss.str();
}

std::string shape_str(const std::vector<int> &shapes) {
    std::ostringstream ss;
    for (int i = 0; i < shapes.size(); ++i) {
        ss << shapes.at(i);
        if (i != shapes.size() - 1) {
            ss << " x ";
        }
    }
    return ss.str();
}

/// 如果图是第一次运行，则根据节点输入operand的形状准备好后续Layer计算中所需要的Tensor
/// 如果图是第二次及以后运行，则检查输入operand的形状和operand中张量的形状是否匹配
void init_operator_input(const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
    // 如果传入的operators为空，则记录错误信息并返回
    if (operators.empty()) {
        LOG(ERROR) << "Operators for init input shapes is empty!";
        return;
    }

    // 遍历所有的operators
    for (const auto &op: operators) {
        // 如果operator的输入operand为空，则跳过当前循环
        if (op->m_input_operands.empty()) {
            continue;
        } else {
            // 获取当前operator的所有输入operand
            const std::map<std::string, std::shared_ptr<RuntimeOperand>> &input_operand_map = op->m_input_operands;
            // 初始化operator的输入空间
            for (const auto &[_, input_operand]: input_operand_map) {
                // 获取当前输入operand的数据类型
                const auto &type = input_operand->m_type;
                // 检查数据类型是否为float32，如果不是则记录错误信息
                CHECK(type == ERuntimeDataType::ERDT_Float32) << "The graph only support float32 yet!";
                // 获取当前输入operand的形状
                const auto &input_operand_shape = input_operand->m_shapes;
                // 获取需要初始化的空间
                auto &input_data = input_operand->m_data;

                // 检查输入形状是否为空
                CHECK(!input_operand_shape.empty());
                // 获取batch大小
                const int32_t batch = input_operand_shape[0];
                // 检查batch大小是否大于等于0，不支持动态batch大小
                CHECK(batch >= 0) << "Dynamic batch size is not supported!";
                // 检查输入形状的维度是否为2, 3, 或4，不支持其他维度
                CHECK(input_operand_shape.size() == 2 ||
                      input_operand_shape.size() == 4 ||
                      input_operand_shape.size() == 3)
                                << "Unsupported tensor shape sizes: " << input_operand_shape.size();
                // 如果输入数据不为空，则检查输入数据的大小是否与batch大小相匹配
                if (!input_data.empty()) {
                    CHECK(input_data.size() == batch);
                } else {
                    // 如果输入数据为空，则根据batch大小初始化输入数据
                    input_data.resize(batch);
                }
            }
        }
    }
}

/// 如果图是第一次运行，则根据节点输出operand的形状准备好后续Layer计算中所需要的Tensor
/// 如果图是第二次及以后运行，则检查输出operand的形状和operand中张量的形状是否匹配
void init_operator_output(const std::vector<pnnx::Operator *> &pnnx_operators,
                          const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
    // 检查传入的pnnx_operators和operators是否为空，并且大小是否一致
    CHECK(!pnnx_operators.empty() && !operators.empty());
    CHECK(pnnx_operators.size() == operators.size());
    for (uint32_t i = 0; i < pnnx_operators.size(); i++) {
        // 得到pnnx原有的输出空间
        const std::vector<pnnx::Operand *> operands = pnnx_operators[i]->outputs;
        // 检查每个节点是否仅有一个输出，目前只支持单输出节点
        CHECK(operands.size() <= 1) << "Only support one node one output yet!";
        if (operands.empty()) {
            continue;
        }
        CHECK(operands.size() == 1) << "Only support one output!";
        // 获取单个输出operand
        pnnx::Operand *operand = operands.front();
        const auto &runtime_op = operators[i];
        // 检查operand是否为空
        CHECK(operand != nullptr) << "Operand output is null";
        // 将operand形状转换为uint32_t向量
        const std::vector<uint32_t> operand_shapes = std::vector<uint32_t>(operand->shape.begin(),
                                                                           operand->shape.end());

        // 获取需要初始化的输出空间
        const auto &output_tensors = runtime_op->m_output_operands;

        // 获取batch大小，目前不支持动态batch大小
        const uint32_t batch = operand_shapes[0];
        CHECK(batch >= 0) << "Dynamic batch size is not supported!";
        // 检查支持的形状大小：2、3或4
        CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 4 ||
              operand_shapes.size() == 3)
                        << "Unsupported shape sizes: " << operand_shapes.size();

        // 如果输出空间未初始化
        if (!output_tensors) {
            // 初始化输出张量
            std::shared_ptr<RuntimeOperand> output_operand =
                    std::make_shared<RuntimeOperand>();
            // 设置输出操作数的形状、类型和名称
            output_operand->m_shapes = operand_shapes;
            output_operand->m_type = ERuntimeDataType::ERDT_Float32;
            output_operand->m_name = operand->name + "_output";
            // 根据batch和形状初始化输出张量
            for (int j = 0; j < batch; ++j) {
                if (operand_shapes.size() == 4) {
                    std::shared_ptr<Tensor> output_tensor = tensor_create(
                            operand_shapes[1], operand_shapes[2], operand_shapes[3]);
                    output_operand->m_data.push_back(output_tensor);
                } else if (operand_shapes.size() == 2) {
                    std::shared_ptr<Tensor> output_tensor = tensor_create(operand_shapes[1]);
                    output_operand->m_data.push_back(output_tensor);
                } else {
                    // 当形状为3时
                    std::shared_ptr<Tensor> output_tensor = tensor_create((uint32_t) operand_shapes[1],
                                                                          (uint32_t) operand_shapes[2]);
                    output_operand->m_data.push_back(output_tensor);
                }
            }
            runtime_op->m_output_operands = std::move(output_operand);
        } else {
            // 如果输出空间已存在，则进行校验
            CHECK(batch == output_tensors->m_data.size());
            CHECK(output_tensors->m_type == ERuntimeDataType::ERDT_Float32);
            CHECK(output_tensors->m_shapes == operand_shapes);
            // 校验并可能重塑每个batch的输出张量形状
            for (uint32_t b = 0; b < batch; ++b) {
                std::shared_ptr<Tensor> output_tensor = output_tensors->m_data.at(b);
                const std::vector<uint32_t> &tensor_shapes = output_tensor->shapes();
                // 根据形状大小进行相应的校验和重塑
                if (operand_shapes.size() == 4) {
                    // 形状为4时的校验和重塑
                    if (tensor_shapes.at(0) != operand_shapes.at(1) ||
                        tensor_shapes.at(1) != operand_shapes.at(2) ||
                        tensor_shapes.at(2) != operand_shapes.at(3)) {
                        LOG(WARNING) << "The shape of tensor do not adapting with output operand";
                        const auto &target_shapes = std::vector<uint32_t>{
                                (uint32_t) operand_shapes.at(1),
                                (uint32_t) operand_shapes.at(2),
                                (uint32_t) operand_shapes.at(3)};
                        output_tensor->reshape(target_shapes);
                    }
                } else if (operand_shapes.size() == 2) {
                    // 形状为2时的校验和重塑
                    if (tensor_shapes.at(0) != 1 ||
                        tensor_shapes.at(1) != operand_shapes.at(1) ||
                        tensor_shapes.at(2) != 1) {
                        LOG(WARNING) << "The shape of tensor do not adapting with output operand";
                        const auto &target_shapes = std::vector<uint32_t>{(uint32_t) operand_shapes.at(1)};
                        output_tensor->reshape(target_shapes);
                    }
                } else {
                    // 形状为3时的校验和重塑
                    if (tensor_shapes.at(0) != 1 ||
                        tensor_shapes.at(1) != operand_shapes.at(1) ||
                        tensor_shapes.at(2) != operand_shapes.at(2)) {
                        LOG(WARNING) << "The shape of tensor do not adapting with output operand";
                        const auto &target_shapes = std::vector<uint32_t>{(uint32_t) operand_shapes.at(1),
                                                                          (uint32_t) operand_shapes.at(2)};
                        output_tensor->reshape(target_shapes);
                    }
                }
            }
        }
    }
}

std::shared_ptr<Tensor> tensor_clone(std::shared_ptr<Tensor> tensor) {
    return std::make_shared<Tensor>(*tensor);
}
