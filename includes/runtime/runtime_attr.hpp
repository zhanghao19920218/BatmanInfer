//
// Created by Mason on 2024/10/15.
//

#ifndef BATMAN_INFER_RUNTIME_ATTR_HPP
#define BATMAN_INFER_RUNTIME_ATTR_HPP

#include <glog/logging.h>
#include <vector>
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace BatmanInfer {
    /**
     * 计算图节点的属性信息
     */
    struct RuntimeAttribute {
        // 节点中的权重参数
        std::vector<float> weight_data;
        // 节点中的形状信息
        std::vector<int> shape;
        // 节点中的数据类型
        RuntimeDataType type = RuntimeDataType::kTypeUnknown;

        /**
         * 从节点中加载权重参数
         * @tparam T 权重类型
         * @param need_clear_weight
         * @return 权重参数数组
         */
        template<class T>
        std::vector<T> get(bool need_clear_weight = true);

        /**
         * 清空权重
         */
        void clearWeight();
    };

    template <class T>
    std::vector<T> RuntimeAttribute::get(bool need_clear_weight) {
        CHECK(!weight_data.empty());
        CHECK(type != RuntimeDataType::kTypeUnknown);
        std::vector<T> weights;
        switch (type) {
            case RuntimeDataType::kTypeFloat32: {
                const bool is_float = std::is_same<T, float>::value;
                CHECK_EQ(is_float, true);
                const uint32_t float_size = sizeof(float);
                CHECK_EQ(weight_data.size() % float_size, 0);
                for (uint32_t i = 0; i < weight_data.size() / float_size; ++i) {
                    float weight = *((float*)weight_data.data() + i);
                    weights.push_back(weight);
                }
                break;
            }
            default: {
                LOG(FATAL) << "Unknown weight data type: " << int(type);
            }
        }
        if (need_clear_weight)
            this->clearWeight();
        return weights;
    }
}


#endif //BATMAN_INFER_RUNTIME_ATTR_HPP
