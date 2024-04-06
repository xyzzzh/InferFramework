//
// Created by xyzzzh on 2024/4/4.
//

#include "parser/ExpressionParser.hpp"

void ExpressionParser::tokenizer(bool retokenize) {
    if (!retokenize && !this->m_tokens.empty()) {
        return;
    }

    CHECK(!this->m_statement.empty()) << "The input statement is empty!";
    this->m_statement.erase(std::remove_if(this->m_statement.begin(), this->m_statement.end(),
                                    [](char c) { return std::isspace(c); }),
                     this->m_statement.end());
    CHECK(!this->m_statement.empty()) << "The input statement is empty!";

    for (int32_t i = 0; i < this->m_statement.size();) {
        char c = this->m_statement.at(i);
        if (c == 'a') {
            CHECK(i + 1 < this->m_statement.size() && this->m_statement.at(i + 1) == 'd')
                            << "Parse add token failed, illegal character: "
                            << this->m_statement.at(i + 1);
            CHECK(i + 2 < this->m_statement.size() && this->m_statement.at(i + 2) == 'd')
                            << "Parse add token failed, illegal character: "
                            << this->m_statement.at(i + 2);
            Token token(ETokenType::ETT_TokenAdd, i, i + 3);
            this->m_tokens.push_back(token);
            std::string token_operation =
                    std::string(this->m_statement.begin() + i, this->m_statement.begin() + i + 3);
            this->m_token_strs.push_back(token_operation);
            i = i + 3;
        } else if (c == 'm') {
            CHECK(i + 1 < this->m_statement.size() && this->m_statement.at(i + 1) == 'u')
                            << "Parse multiply token failed, illegal character: "
                            << this->m_statement.at(i + 1);
            CHECK(i + 2 < this->m_statement.size() && this->m_statement.at(i + 2) == 'l')
                            << "Parse multiply token failed, illegal character: "
                            << this->m_statement.at(i + 2);
            Token token(ETokenType::ETT_TokenMul, i, i + 3);
            this->m_tokens.push_back(token);
            std::string token_operation =
                    std::string(this->m_statement.begin() + i, this->m_statement.begin() + i + 3);
            this->m_token_strs.push_back(token_operation);
            i = i + 3;
        } else if (c == '@') {
            CHECK(i + 1 < this->m_statement.size() && std::isdigit(this->m_statement.at(i + 1)))
                            << "Parse number token failed, illegal character: "
                            << this->m_statement.at(i + 1);
            int32_t j = i + 1;
            for (; j < this->m_statement.size(); ++j) {
                if (!std::isdigit(this->m_statement.at(j))) {
                    break;
                }
            }
            Token token(ETokenType::ETT_TokenInputNumber, i, j);
            CHECK(token.start_pos < token.end_pos);
            this->m_tokens.push_back(token);
            std::string token_input_number =
                    std::string(this->m_statement.begin() + i, this->m_statement.begin() + j);
            this->m_token_strs.push_back(token_input_number);
            i = j;
        } else if (c == ',') {
            Token token(ETokenType::ETT_TokenComma, i, i + 1);
            this->m_tokens.push_back(token);
            std::string token_comma =
                    std::string(this->m_statement.begin() + i, this->m_statement.begin() + i + 1);
            this->m_token_strs.push_back(token_comma);
            i += 1;
        } else if (c == '(') {
            Token token(ETokenType::ETT_TokenLeftBracket, i, i + 1);
            this->m_tokens.push_back(token);
            std::string token_left_bracket =
                    std::string(this->m_statement.begin() + i, this->m_statement.begin() + i + 1);
            this->m_token_strs.push_back(token_left_bracket);
            i += 1;
        } else if (c == ')') {
            Token token(ETokenType::ETT_TokenRightBracket, i, i + 1);
            this->m_tokens.push_back(token);
            std::string token_right_bracket =
                    std::string(this->m_statement.begin() + i, this->m_statement.begin() + i + 1);
            this->m_token_strs.push_back(token_right_bracket);
            i += 1;
        } else {
            LOG(FATAL) << "Unknown  illegal character: " << c;
        }
    }
}

std::vector<std::shared_ptr<TokenNode>> ExpressionParser::generate() {
    if (this->m_tokens.empty()) {
        this->tokenizer(true);
    }
    int index = 0;
    std::shared_ptr<TokenNode> root = this->_generate(index);
    CHECK(root != nullptr);
    CHECK(index == this->m_tokens.size() - 1);
    std::vector<std::shared_ptr<TokenNode>> reverse_polish;
    std::function<void(std::shared_ptr<TokenNode>)> dfs = [&](std::shared_ptr<TokenNode> root_node) {
        if (root_node != nullptr) {
            dfs(root_node->left);
            dfs(root_node->right);
            reverse_polish.push_back(root_node);
        }
    };
    dfs(root);
    return reverse_polish;
}

const std::vector<Token> &ExpressionParser::tokens() const {
    return this->m_tokens;
}

const std::vector<std::string> &ExpressionParser::token_strs() const {
    return this->m_token_strs;
}

std::shared_ptr<TokenNode> ExpressionParser::_generate(int32_t &index) {
    CHECK(index < this->m_tokens.size());
    const auto current_token = this->m_tokens.at(index);
    CHECK(current_token.token_type == ETokenType::ETT_TokenInputNumber ||
          current_token.token_type == ETokenType::ETT_TokenAdd ||
          current_token.token_type == ETokenType::ETT_TokenMul);
    if (current_token.token_type == ETokenType::ETT_TokenInputNumber) {
        uint32_t start_pos = current_token.start_pos + 1;
        uint32_t end_pos = current_token.end_pos;
        CHECK(end_pos > start_pos || end_pos <= this->m_statement.length())
                        << "Current token has a wrong length";
        const std::string &str_number =
                std::string(this->m_statement.begin() + start_pos,
                            this->m_statement.begin() + end_pos);
        return std::make_shared<TokenNode>(std::stoi(str_number), nullptr, nullptr);

    } else if (current_token.token_type == ETokenType::ETT_TokenMul ||
               current_token.token_type == ETokenType::ETT_TokenAdd) {
        std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
        current_node->num_index = int(current_token.token_type);

        index += 1;
        CHECK(index < this->m_tokens.size()) << "Missing left bracket!";
        CHECK(this->m_tokens.at(index).token_type == ETokenType::ETT_TokenLeftBracket);

        index += 1;
        CHECK(index < this->m_tokens.size()) << "Missing correspond left token!";
        const auto left_token = this->m_tokens.at(index);

        if (left_token.token_type == ETokenType::ETT_TokenInputNumber ||
            left_token.token_type == ETokenType::ETT_TokenAdd ||
            left_token.token_type == ETokenType::ETT_TokenMul) {
            current_node->left = _generate(index);
        } else {
            LOG(FATAL) << "Unknown token type: " << int(left_token.token_type);
        }

        index += 1;
        CHECK(index < this->m_tokens.size()) << "Missing comma!";
        CHECK(this->m_tokens.at(index).token_type == ETokenType::ETT_TokenComma);

        index += 1;
        CHECK(index < this->m_tokens.size()) << "Missing correspond right token!";
        const auto right_token = this->m_tokens.at(index);
        if (right_token.token_type == ETokenType::ETT_TokenInputNumber ||
            right_token.token_type == ETokenType::ETT_TokenAdd ||
            right_token.token_type == ETokenType::ETT_TokenMul) {
            current_node->right = _generate(index);
        } else {
            LOG(FATAL) << "Unknown token type: " << int(right_token.token_type);
        }

        index += 1;
        CHECK(index < this->m_tokens.size()) << "Missing right bracket!";
        CHECK(this->m_tokens.at(index).token_type == ETokenType::ETT_TokenRightBracket);
        return current_node;
    } else {
        LOG(FATAL) << "Unknown token type: " << int(current_token.token_type);
    }
}
