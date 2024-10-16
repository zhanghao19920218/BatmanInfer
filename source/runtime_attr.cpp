//
// Created by Mason on 2024/10/15.
//

#include <runtime/runtime_attr.hpp>

namespace BatmanInfer {
    void RuntimeAttribute::clearWeight() {
        if (!this->weight_data.empty()) {
            std::vector<float> tmp = std::vector<float>();
            this->weight_data.swap(tmp);
        }
    }
}