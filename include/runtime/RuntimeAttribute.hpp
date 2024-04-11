//
// Created by xyzzzh on 2024/3/30.
//

#ifndef INFERFRAMEWORK_RUNTIMEATTRIBUTE_HPP
#define INFERFRAMEWORK_RUNTIMEATTRIBUTE_HPP

#include "Common.hpp"

// 计算图节点的属性信息结构体。
struct RuntimeAttribute {
    // 节点中的权重参数，以char形式存储。
    std::vector<char> m_weight_data;

    // 节点的形状信息，存储为一个无符号整数向量。
    std::vector<uint32_t> m_shapes;

    // 节点中的数据类型，使用枚举ERuntimeDataType表示，默认为ERDT_Unknown。
    ERuntimeDataType m_type = ERuntimeDataType::ERDT_Unknown;

    // 从节点的权重数据中提取权重，并将其转换为指定的数据类型T的向量。
    // 如果need_clear_weight为true，则在提取权重数据后清除原始的权重数据。
    template<typename T>
    std::vector<T> get_weight_data(bool need_clear_weight = true) {
        // 确保权重数据不为空。
        CHECK(!this->m_weight_data.empty());

        // 确保数据类型不是未知的。
        CHECK(this->m_type != ERuntimeDataType::ERDT_Unknown);

        std::vector<T> weight; // 用于存储转换后的权重数据。

        // 根据数据类型处理权重数据。
        switch (this->m_type) {
            case ERuntimeDataType::ERDT_Float32: { // 如果数据类型为float32。
                // 验证模板类型T是否为float。
                const bool is_float = std::is_same_v<T, float>;
                CHECK(is_float);

                // 确保权重数据大小能整除float的大小。
                const uint32_t float_size = sizeof(float);
                const uint32_t weight_size = this->m_weight_data.size();
                CHECK(weight_size % float_size == 0);

                // 将权重数据从char数组转换为float数组。
                auto *start_ptr = reinterpret_cast<const float *>(this->m_weight_data.data());
                auto *end_ptr = start_ptr + weight_size / float_size;
                weight.insert(weight.end(), start_ptr, end_ptr);
                break;
            }
            default: { // 如果数据类型未知。
                LOG(FATAL) << "Unknown weight data type" << int(this->m_type);
            }
        }

        // 如果需要，则清除权重数据。
        if (need_clear_weight) {
            this->clear_weight();
        }
        return weight;
    }

    // 清除权重数据。
    void clear_weight() {
        if (!this->m_weight_data.empty()) {
            m_weight_data.clear();
        }
    }
};


#endif //INFERFRAMEWORK_RUNTIMEATTRIBUTE_HPP
