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
    using Creator = std::function<EParseParameterAttrStatus(const std::shared_ptr<RuntimeOperator> &,
                                                            std::shared_ptr<Layer> &)>;
    using CreateRegistry = std::map<std::string, Creator>;


    // 向注册表注册算子
    static void register_creator(const std::string &layer_type, const Creator &creator);

    // 通过算子参数op来初始化Layer
    static std::shared_ptr<Layer> create_layer(const std::shared_ptr<RuntimeOperator> &op);

    // 返回算子的注册表
    static CreateRegistry &get_registry();

    // 返回所有已被注册算子的类型
    static std::vector<std::string> layer_types();

    static void layer_registerer_wrapper(const std::string &layer_type, const Creator &creator) {
        register_creator(layer_type, creator);
    }

    static bool compare_CreateRegistry(const CreateRegistry& c1, const CreateRegistry& c2);
};


#endif //INFERFRAMEWORK_LAYERREGISTERER_HPP
