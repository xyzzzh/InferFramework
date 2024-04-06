//
// Created by xyzzzh on 2024/4/4.
//

#ifndef INFERFRAMEWORK_EXPRESSIONPARSER_HPP
#define INFERFRAMEWORK_EXPRESSIONPARSER_HPP

#include "Common.hpp"

struct Token {
    ETokenType token_type = ETokenType::ETT_TokenUnknown;
    uint32_t start_pos = 0;
    uint32_t end_pos = 0;

    Token(ETokenType _token_type, uint32_t _start_pos, uint32_t _end_pos) :
            token_type(_token_type), start_pos(_start_pos), end_pos(_end_pos) {}
};

struct TokenNode {
    int32_t num_index = -1;
    std::shared_ptr<TokenNode> left = nullptr;
    std::shared_ptr<TokenNode> right = nullptr;

    TokenNode() = default;

    TokenNode(int32_t _num_index, std::shared_ptr<TokenNode> _left, std::shared_ptr<TokenNode> _right) :
            num_index(_num_index), left(_left), right(_right) {}
};

class ExpressionParser {
public:
    explicit ExpressionParser(std::string _statement) : m_statement(std::move(_statement)) {}

    // 词法分析
    // retokenize: 是否需要重新进行语法分析
    void tokenizer(bool retokenize = false);

    // 语法分析
    std::vector<std::shared_ptr<TokenNode>> generate();

    // 返回词法分析的结果
    const std::vector<Token>& tokens() const;

    // 返回词语字符串
    const std::vector<std::string>& token_strs() const;

    std::shared_ptr<TokenNode> _generate(int32_t& index);

private:
    std::vector<Token> m_tokens;            /// 被分割的词语数组
    std::vector<std::string> m_token_strs;  /// 被分割的字符串数组
    std::string m_statement;                /// 待分割的表达式
};


#endif //INFERFRAMEWORK_EXPRESSIONPARSER_HPP
