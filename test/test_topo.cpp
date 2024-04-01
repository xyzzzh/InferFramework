//
// Created by xyzzzh on 2024/4/1.
//

#include "Common.hpp"
#include "Utils.hpp"
#include "runtime/RuntimeGraph.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

TEST(test_ir, topo) {
    
    std::string bin_path("model_file/resnet18_batch1.pnnx.bin");
    std::string param_path("model_file/resnet18_batch1.param");
    RuntimeGraph graph(param_path, bin_path);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success, true);
    graph.build("pnnx_input_0", "pnnx_output_0");
    const auto &topo_queues = graph.get_topo_queues();

    int index = 0;
    for (const auto &operator_ : topo_queues) {
        LOG(INFO) << "Index: " << index << " Type: " << operator_->m_type
                  << " Name: " << operator_->m_name;
        index += 1;
    }
}

TEST(test_ir, build_output_ops) {
    
    std::string bin_path("model_file/simple_ops.pnnx.bin");
    std::string param_path("model_file/simple_ops.pnnx.param");
    RuntimeGraph graph(param_path, bin_path);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success, true);
    graph.build("pnnx_input_0", "pnnx_output_0");
    const auto &topo_queues = graph.get_topo_queues();

    int index = 0;
    for (const auto &operator_ : topo_queues) {
        LOG(INFO) << "Index: " << index << " Name: " << operator_->m_name;
        index += 1;
    }
}

TEST(test_ir, build_output_ops2) {
    
    std::string bin_path("model_file/simple_ops.pnnx.bin");
    std::string param_path("model_file/simple_ops.pnnx.param");
    RuntimeGraph graph(param_path, bin_path);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success, true);
    graph.build("pnnx_input_0", "pnnx_output_0");
    const auto &topo_queues = graph.get_topo_queues();

    int index = 0;
    for (const auto &operator_ : topo_queues) {
        LOG(INFO) << "operator name: " << operator_->m_name;
        for (const auto &pair : operator_->m_output_operators) {
            LOG(INFO) << "output: " << pair.first;
        }
        LOG(INFO) << "-------------------------";
        index += 1;
    }
}

TEST(test_ir, build1_status) {
    
    std::string bin_path("model_file/simple_ops.pnnx.bin");
    std::string param_path("model_file/simple_ops.pnnx.param");
    RuntimeGraph graph(param_path, bin_path);
    ASSERT_EQ(int(graph.state()), -2);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.state()), -1);
    graph.build("pnnx_input_0", "pnnx_output_0");
    ASSERT_EQ(int(graph.state()), 0);
}

TEST(test_ir, build1_output_tensors) {
    
    std::string bin_path("model_file/simple_ops2.pnnx.bin");
    std::string param_path("model_file/simple_ops2.pnnx.param");
    RuntimeGraph graph(param_path, bin_path);
    ASSERT_EQ(int(graph.state()), -2);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success, true);
    ASSERT_EQ(int(graph.state()), -1);
    graph.build("pnnx_input_0", "pnnx_output_0");
    ASSERT_EQ(int(graph.state()), 0);

    const auto &ops = graph.operators();
    for (const auto &op : ops) {
        LOG(INFO) << op->m_name;
        // 打印op输出空间的张量
        const auto &operand = op->m_output_operands;
        if (!operand || operand->m_data.empty()) {
            continue;
        }
        const uint32_t batch_size = operand->m_data.size();
        LOG(INFO) << "batch: " << batch_size;

        for (uint32_t i = 0; i < batch_size; ++i) {
            const auto &data = operand->m_data.at(i);
            LOG(INFO) << "channel: " << data->channels()
                      << " height: " << data->rows() << " cols: " << data->cols();
        }
    }
}