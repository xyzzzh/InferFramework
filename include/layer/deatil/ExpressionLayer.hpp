//
// Created by xyzzzh on 2024/4/6.
//

#ifndef INFERFRAMEWORK_EXPRESSIONLAYER_HPP
#define INFERFRAMEWORK_EXPRESSIONLAYER_HPP

#include "Common.hpp"
#include "runtime/RuntimeGraph.hpp"
#include "layer/abstract/LayerRegisterer.hpp"
#include "layer/abstract/NonParamLayer.hpp"
#include "parser/ExpressionParser.hpp"

class ExpressionLayer : public NonParamLayer {
public:
    explicit ExpressionLayer(std::string statement);

    EInferStatus forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                         std::vector<std::shared_ptr<Tensor>> &outputs) override;

    static EParseParameterAttrStatus get_instance(const std::shared_ptr<RuntimeOperator> &op,
                                                  std::shared_ptr<Layer> &expression_layer);

private:
    std::string m_statement;
    std::unique_ptr<ExpressionParser> m_parser;
};


#endif //INFERFRAMEWORK_EXPRESSIONLAYER_HPP
