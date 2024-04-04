//
// Created by xyzzzh on 2024/3/30.
//

#ifndef INFERFRAMEWORK_RUNTIMEATTRIBUTE_HPP
#define INFERFRAMEWORK_RUNTIMEATTRIBUTE_HPP

#include "Common.hpp"

// 计算图节点的属性信息
struct RuntimeAttribute {
    std::vector<char> m_weight_data;                             // 节点中的权重参数
    std::vector<uint32_t> m_shapes;                              // 节点中的形状信息
    ERuntimeDataType m_type = ERuntimeDataType::ERDT_Unknown;    // 节点中的数据类型

    // 从节点中加载权重参数
    template<typename T>
    std::vector<T> get_weight_data(bool need_clear_weight = true) {
        CHECK(!this->m_weight_data.empty());
        CHECK(this->m_type != ERuntimeDataType::ERDT_Unknown);

        std::vector<T> weight;

        switch (this->m_type) {
            // float32
            case ERuntimeDataType::ERDT_Float32: {
                // 检查是否为float
                const bool is_float = std::is_same_v<T, float>;
                CHECK(is_float);
                // 检查是否为float32
                const uint32_t float_size = sizeof(float);
                const uint32_t weight_size = this->m_weight_data.size();
                CHECK(weight_size % float_size == 0);

                auto *start_ptr = reinterpret_cast<const float *>(this->m_weight_data.data());
                auto *end_ptr = start_ptr + weight_size / float_size;
                weight.insert(weight.end(), start_ptr, end_ptr);

                break;
            }
            default: {
                LOG(FATAL) << "Unknown weight data type" << int(this->m_type);
            }
        }

        if (need_clear_weight) {
            this->clear_weight();
        }
        return weight;
    }

    void clear_weight() {
        if (!this->m_weight_data.empty()) {
            m_weight_data.clear();
        }
    }

};


#endif //INFERFRAMEWORK_RUNTIMEATTRIBUTE_HPP
