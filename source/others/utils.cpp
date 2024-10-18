//
// Created by Mason on 2024/10/13.
//
#include <others/utils.hpp>

namespace BatmanInfer {
    std::string ShapeStr(const std::vector<int> &shapes) {
        std::ostringstream ss;
        for (int i = 0; i < shapes.size(); ++i) {
            ss << shapes.at(i);
            if (i != shapes.size() - 1) {
                ss << " x ";
            }
        }
        return ss.str();
    }
}