//
// Created by xyzzzh on 2024/4/2.
//

#include "layer/abstract/LayerRegisterer.hpp"

std::shared_ptr<Layer> LayerRegisterer::create_layer(const std::shared_ptr<RuntimeOperator> &op) {
    CreateRegistry &registry = get_registry();
    const std::string &layer_type = op->m_type;
    LOG_IF(FATAL, registry.count(layer_type) <= 0) << "Can not find the layer type: " << layer_type;
    const auto &creator = registry.find(layer_type)->second;
    LOG_IF(FATAL, !creator) << "Layer creator is empty!";
    std::shared_ptr<Layer> layer;
    const auto &status = creator(op, layer);
    LOG_IF(FATAL, status != EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess)
                    << "Create the layer: " << layer_type
                    << " failed, error code: " << int(status);
    return layer;
}

LayerRegisterer::CreateRegistry &LayerRegisterer::get_registry() {
    static CreateRegistry *registry = new CreateRegistry();
    CHECK(registry != nullptr) << "Global layer register init failed!";
    return *registry;
}

void LayerRegisterer::register_creator(const std::string &layer_type, const Creator &creator) {
    CHECK(creator != nullptr);
    CreateRegistry &registry = get_registry();
    CHECK(registry.count(layer_type) == 0) << "Layer type: " << layer_type << " has already registered!";
    registry.insert({layer_type, creator});
}

std::vector<std::string> LayerRegisterer::layer_types() {
    std::vector<std::string> layer_types;
    static CreateRegistry &registry = get_registry();
    for (const auto &[layer_type, _]: registry) {
        layer_types.push_back(layer_type);
    }
    return layer_types;
}

bool LayerRegisterer::compare_CreateRegistry(const CreateRegistry &c1, const CreateRegistry &c2) {
    if (c1.size() != c2.size()) {
        return false;
    }

    for (const auto &pair: c1) {
        if (c2.find(pair.first) == c2.end()) {
            return false;
        }
    }

    return true;
}
