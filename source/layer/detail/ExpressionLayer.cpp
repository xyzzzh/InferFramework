//
// Created by xyzzzh on 2024/4/6.
//

#include "layer/deatil/ExpressionLayer.hpp"

ExpressionLayer::ExpressionLayer(std::string statement) : NonParamLayer("Expression"),
                                                          m_statement(std::move(statement)) {
    this->m_parser = std::make_unique<ExpressionParser>(statement);

}

EInferStatus ExpressionLayer::forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                      std::vector<std::shared_ptr<Tensor>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the expression layer is empty";
        return EInferStatus::EIS_InferFailedInputEmpty;
    }

    if (outputs.empty()) {
        LOG(ERROR) << "The output tensor array in the expression layer is empty";
        return EInferStatus::EIS_InferFailedOutputEmpty;
    }

    CHECK(this->m_parser != nullptr)
                    << "The parser in the expression layer is null!";
    this->m_parser->tokenizer(false);
    const auto &expressions = this->m_parser->tokens();
    CHECK(!expressions.empty())
                    << "The expression parser failed to parse " << this->m_statement;

    for (uint32_t i = 0; i < inputs.size(); ++i) {
        const std::shared_ptr<Tensor> &input_data = inputs.at(i);
        if (input_data == nullptr || input_data->empty()) {
            LOG(ERROR) << "The input tensor array in the expression layer has an "
                          "empty tensor "
                       << i << "th";
            return EInferStatus::EIS_InferFailedInputEmpty;
        }
    }

    const uint32_t batch_size = outputs.size();
    for (uint32_t i = 0; i < batch_size; ++i) {
        if (outputs.at(i) == nullptr || outputs.at(i)->empty()) {
            DLOG(ERROR) << "The output tensor array in the expression layer has an "
                           "empty tensor "
                        << i << "th";
            return EInferStatus::EIS_InferFailedOutputEmpty;
        }
        outputs.at(i)->fill(0.f);
    }

    std::stack<std::vector<std::shared_ptr<Tensor>>> op_stack;
    const std::vector<std::shared_ptr<TokenNode>> &token_nodes =
            this->m_parser->generate();
    for (const auto &token_node: token_nodes) {
        if (token_node->num_index >= 0) {
            // process operator
            uint32_t start_pos = token_node->num_index * batch_size;
            std::vector<std::shared_ptr<Tensor>> input_token_nodes;
            for (uint32_t i = 0; i < batch_size; ++i) {
                CHECK(i + start_pos < inputs.size())
                                << "The " << i
                                << "th operand doesn't have appropriate number of tensors";
                input_token_nodes.push_back(inputs.at(i + start_pos));
            }
            op_stack.push(input_token_nodes);
        } else {
            // process operation
            const int32_t op = token_node->num_index;
            if (op != int(ETokenType::ETT_TokenAdd) && op != int(ETokenType::ETT_TokenMul)) {
                LOG(FATAL) << "Unknown operator type: " << op;
            }
            CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
            std::vector<std::shared_ptr<Tensor>> input_node1 = op_stack.top();

            CHECK(input_node1.size() == batch_size)
                            << "The first operand doesn't have appropriate number of tensors, "
                               "which need "
                            << batch_size;
            op_stack.pop();

            std::vector<std::shared_ptr<Tensor>> input_node2 = op_stack.top();
            CHECK(input_node2.size() == batch_size)
                            << "The second operand doesn't have appropriate number of tensors, "
                               "which need "
                            << batch_size;
            op_stack.pop();

            std::vector<std::shared_ptr<Tensor>> output_token_nodes(
                    batch_size);
            for (uint32_t i = 0; i < batch_size; ++i) {
                // do execution
                if (op == int(ETokenType::ETT_TokenAdd)) {
                    output_token_nodes.at(i) =
                            tensor_add(input_node1.at(i), input_node2.at(i));
                } else if (op == int(ETokenType::ETT_TokenMul)) {
                    output_token_nodes.at(i) =
                            tensor_multiply(input_node1.at(i), input_node2.at(i));
                } else {
                    LOG(FATAL) << "Unknown operator type: " << op;
                }
            }
            op_stack.push(output_token_nodes);
        }
    }

    CHECK(op_stack.size() == 1)
                    << "The expression has more than one output operand!";
    std::vector<std::shared_ptr<Tensor>> output_node = op_stack.top();
    op_stack.pop();
    for (int i = 0; i < batch_size; ++i) {
        CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
        CHECK(outputs.at(i)->shapes() == output_node.at(i)->shapes());
        outputs.at(i) = output_node.at(i);
    }
    return EInferStatus::EIS_InferSuccess;
}

EParseParameterAttrStatus
ExpressionLayer::get_instance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &expression_layer) {
    CHECK(op != nullptr) << "Expression operator is nullptr";
    const auto &params = op->m_params;
    if (params.find("expr") == params.end()) {
        return EParseParameterAttrStatus::EPPAS_ParameterMissingExpr;
    }

    auto m_statementparam = std::dynamic_pointer_cast<RuntimeParameterString>(params.at("expr"));
    if (m_statementparam == nullptr || m_statementparam->type != ERuntimeParameterType::ERPT_ParameterString) {
        LOG(ERROR) << "Can not find the expression parameter";
        return EParseParameterAttrStatus::EPPAS_ParameterMissingExpr;
    }

    expression_layer = std::make_shared<ExpressionLayer>(m_statementparam->value);
    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;

}

LayerRegistererWrapper expression_get_instance("pnnx.Expression", ExpressionLayer::get_instance);