//
// Created by Mason on 2024/10/13.
//
#include <others/utils.hpp>

namespace BatmanInfer {
    std::string ShapeStr(const onnx::TypeProto& type_proto) {
        if (!type_proto.has_tensor_type())
            return "UnknownType";
        const auto& tensor_type = type_proto.tensor_type();
        if (!tensor_type.has_shape())
            return "UnknownShape";
        const auto& shape = tensor_type.shape();
        std::string shape_str = "[";
        for (int i = 0; i < shape.dim_size(); ++i) {
            const auto& dim = shape.dim(i);
            if (dim.has_dim_value()) {
                shape_str += std::to_string(dim.dim_value());
            } else if (dim.has_dim_param()) {
                shape_str += dim.dim_param(); // 参数维度
            } else {
                shape_str += "?"; // 未知维度
            }
            if (i != shape.dim_size() -1) {
                shape_str += ", ";
            }
        }
        shape_str += "]";
        return shape_str;
    }
}