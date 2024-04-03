//
// Created by xyzzzh on 2024/4/2.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "layer/abstract/LayerRegisterer.hpp"

static LayerRegisterer::CreateRegistry *RegistryGlobal() {
    static LayerRegisterer::CreateRegistry *kRegistry = new LayerRegisterer::CreateRegistry();
    CHECK(kRegistry != nullptr) << "Global layer register init failed!";
    return kRegistry;
}

TEST(test_registry, registry1) {

    LayerRegisterer::CreateRegistry *registry1 = RegistryGlobal();
    LayerRegisterer::CreateRegistry *registry2 = RegistryGlobal();

    LayerRegisterer::CreateRegistry *registry3 = RegistryGlobal();
    LayerRegisterer::CreateRegistry *registry4 = RegistryGlobal();
    float *a = new float{3};
    float *b = new float{4};
    ASSERT_EQ(registry1, registry2);
}

EParseParameterAttrStatus MyTestCreator(
        const std::shared_ptr<RuntimeOperator> &op,
        std::shared_ptr<Layer> &layer) {

    layer = std::make_shared<Layer>("test_layer");
    return EParseParameterAttrStatus::EPPAS_ParameterAttrParseSuccess;
}

TEST(test_registry, registry2) {
    LayerRegisterer::CreateRegistry registry1 = LayerRegisterer::get_registry();
    LayerRegisterer::CreateRegistry registry2 = LayerRegisterer::get_registry();
    ASSERT_TRUE(LayerRegisterer::compare_CreateRegistry(registry1, registry2));
    auto default_size = LayerRegisterer::get_registry().size();
    LayerRegisterer::register_creator("test_type", MyTestCreator);
    LayerRegisterer::CreateRegistry registry3 = LayerRegisterer::get_registry();
    ASSERT_EQ(registry3.size(), default_size+1);
    ASSERT_NE(registry3.find("test_type"), registry3.end());
}

TEST(test_registry, create_layer) {
    // 注册了一个test_type_1算子
    LayerRegisterer::register_creator("test_type_1", MyTestCreator);
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->m_type = "test_type_1";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegisterer::create_layer(op);
    ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_util) {
    LayerRegistererWrapper ReluGetInstance("test_type_2", MyTestCreator);
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->m_type = "test_type_2";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegisterer::create_layer(op);
    ASSERT_NE(layer, nullptr);
}


TEST(test_registry, create_layer_reluforward) {
    std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
    op->m_type = "nn.ReLU";
    std::shared_ptr<Layer> layer;
    ASSERT_EQ(layer, nullptr);
    layer = LayerRegisterer::create_layer(op);
    ASSERT_NE(layer, nullptr);

    std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>(3, 4, 4);
    input_tensor->rand();
    input_tensor->data() -= 0.5f;

    LOG(INFO) << input_tensor->data();

    std::vector<std::shared_ptr<Tensor>> inputs(1);
    std::vector<std::shared_ptr<Tensor>> outputs(1);
    inputs.at(0) = input_tensor;
    layer->forward(inputs, outputs);

    for (const auto &output : outputs) {
        output->show();
    }
}