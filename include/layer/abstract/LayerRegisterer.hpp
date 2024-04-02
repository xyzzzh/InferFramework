//
// Created by xyzzzh on 2024/4/2.
//

#ifndef INFERFRAMEWORK_LAYERREGISTERER_HPP
#define INFERFRAMEWORK_LAYERREGISTERER_HPP

#include "Common.hpp"
#include "runtime/RuntimeOperator.hpp"
#include "layer/abstract/Layer.hpp"

class LayerRegisterer {
public:
    // typedef EParseParameterAttrStatus (*Creator)(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer> &layer);
    // typedef std::map<std::string, Creator> CreateRegistry;
    using Creator = std::function<EParseParameterAttrStatus(const std::shared_ptr<RuntimeOperator> &,
                                                            std::shared_ptr<Layer> &)>;
    using CreateRegistry = std::map<std::string, Creator>;


    // 向注册表注册算子
    static void RegisterCreator(const std::string &layer_type, const Creator &creator);

    // 通过算子参数op来初始化Layer
    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

    // 返回算子的注册表
    static CreateRegistry &Registry();

    // 返回所有已被注册算子的类型
    static std::vector<std::string> layer_types();

    inline void LayerRegistererWrapper(const std::string &layer_type, const Creator &creator) {
        RegisterCreator(layer_type, creator);
    }
};


#endif //INFERFRAMEWORK_LAYERREGISTERER_HPP
