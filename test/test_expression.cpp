//
// Created by xyzzzh on 2024/4/6.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "Common.hpp"
#include "runtime/RuntimeParameter.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/deatil/ConvLayer.hpp"
#include "parser/ExpressionParser.hpp"
#include "layer/deatil/ExpressionLayer.hpp"

TEST(test_parser, tokenizer) {

    const std::string &str = "add(@0,mul(@1,@2))";
    ExpressionParser parser(str);
    parser.tokenizer();
    const auto &tokens = parser.tokens();
    ASSERT_EQ(tokens.empty(), false);

    const auto &token_strs = parser.token_strs();
    ASSERT_EQ(token_strs.at(0), "add");
    ASSERT_EQ(tokens.at(0).token_type, ETokenType::ETT_TokenAdd);

    ASSERT_EQ(token_strs.at(1), "(");
    ASSERT_EQ(tokens.at(1).token_type, ETokenType::ETT_TokenLeftBracket);

    ASSERT_EQ(token_strs.at(2), "@0");
    ASSERT_EQ(tokens.at(2).token_type, ETokenType::ETT_TokenInputNumber);

    ASSERT_EQ(token_strs.at(3), ",");
    ASSERT_EQ(tokens.at(3).token_type, ETokenType::ETT_TokenComma);

    ASSERT_EQ(token_strs.at(4), "mul");
    ASSERT_EQ(tokens.at(4).token_type, ETokenType::ETT_TokenMul);

    ASSERT_EQ(token_strs.at(5), "(");
    ASSERT_EQ(tokens.at(5).token_type, ETokenType::ETT_TokenLeftBracket);

    ASSERT_EQ(token_strs.at(6), "@1");
    ASSERT_EQ(tokens.at(6).token_type, ETokenType::ETT_TokenInputNumber);

    ASSERT_EQ(token_strs.at(7), ",");
    ASSERT_EQ(tokens.at(7).token_type, ETokenType::ETT_TokenComma);

    ASSERT_EQ(token_strs.at(8), "@2");
    ASSERT_EQ(tokens.at(8).token_type, ETokenType::ETT_TokenInputNumber);

    ASSERT_EQ(token_strs.at(9), ")");
    ASSERT_EQ(tokens.at(9).token_type, ETokenType::ETT_TokenRightBracket);

    ASSERT_EQ(token_strs.at(10), ")");
    ASSERT_EQ(tokens.at(10).token_type, ETokenType::ETT_TokenRightBracket);
}

TEST(test_parser, generate1) {

    const std::string &str = "add(@0,@1)";
    ExpressionParser parser(str);
    parser.tokenizer();
    int index = 0; // 从0位置开始构建语法树
// 抽象语法树:
//
//    add
//    /  \
  //  @0    @1

    const auto &node = parser._generate(index);
    ASSERT_EQ(node->num_index, int(ETokenType::ETT_TokenAdd));
    ASSERT_EQ(node->left->num_index, 0);
    ASSERT_EQ(node->right->num_index, 1);
}

TEST(test_parser, generate2) {

    const std::string &str = "add(mul(@0,@1),@2)";
    ExpressionParser parser(str);
    parser.tokenizer();
    int index = 0; // 从0位置开始构建语法树
// 抽象语法树:
//
//       add
//       /  \
  //     mul   @2
//    /   \
  //  @0    @1

    const auto &node = parser._generate(index);
    ASSERT_EQ(node->num_index, int(ETokenType::ETT_TokenAdd));
    ASSERT_EQ(node->left->num_index, int(ETokenType::ETT_TokenMul));
    ASSERT_EQ(node->left->left->num_index, 0);
    ASSERT_EQ(node->left->right->num_index, 1);

    ASSERT_EQ(node->right->num_index, 2);
}

TEST(test_parser, reverse_polish) {

    const std::string &str = "add(mul(@0,@1),@2)";
    ExpressionParser parser(str);
    parser.tokenizer();
// 抽象语法树:
//
//       add
//       /  \
  //     mul   @2
//    /   \
  //  @0    @1

    const auto &vec = parser.generate();
    for (const auto &item: vec) {
        if (item->num_index == -5) {
            LOG(INFO) << "Mul";
        } else if (item->num_index == -6) {
            LOG(INFO) << "Add";
        } else {
            LOG(INFO) << item->num_index;
        }
    }
}

TEST(test_expression, complex1) {

    const std::string &str = "mul(@2,add(@0,@1))";
    ExpressionLayer layer(str);
    std::shared_ptr<Tensor> input1 =
            std::make_shared<Tensor>(3, 224, 224);
    input1->fill(2.f);
    std::shared_ptr<Tensor> input2 =
            std::make_shared<Tensor>(3, 224, 224);
    input2->fill(3.f);

    std::shared_ptr<Tensor> input3 =
            std::make_shared<Tensor>(3, 224, 224);
    input3->fill(4.f);

    std::vector<std::shared_ptr<Tensor>> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);

    std::vector<std::shared_ptr<Tensor>> outputs(1);
    outputs.at(0) = std::make_shared<Tensor>(3, 224, 224);
    const auto status = layer.forward(inputs, outputs);
    ASSERT_EQ(status, EInferStatus::EIS_InferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    std::shared_ptr<Tensor> output2 =
            std::make_shared<Tensor>(3, 224, 224);
    output2->fill(20.f);
    std::shared_ptr<Tensor> output1 = outputs.front();

    ASSERT_TRUE(
            arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}