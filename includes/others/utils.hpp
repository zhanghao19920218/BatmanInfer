//
// Created by Mason on 2024/10/13.
//

#ifndef BATMANINFER_UTILS_H
#define BATMANINFER_UTILS_H
#include <string>
#include <onnx/onnx_pb.h>

namespace BatmanInfer {
    std::string ShapeStr(const onnx::TypeProto& type_proto);
}

#endif //BATMANINFER_UTILS_H
