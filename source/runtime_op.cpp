//
// Created by Mason on 2024/10/16.
//

#include <runtime/runtime_op.hpp>

namespace BatmanInfer {
    RuntimeOperator::~RuntimeOperator() {
        for (auto& [_, param] : this->params) {
            if (param != nullptr) {
                delete param;
                param = nullptr;
            }
        }
    }
}