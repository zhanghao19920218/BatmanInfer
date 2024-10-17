//
// Created by Mason on 2024/10/16.
//

#include "data/tensor.hpp"
#include "runtime/ir.h"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

TEST(test_ir, onnx_graph_all) {
    using namespace BatmanInfer;
    std::string model_path("./model/model.onnx");
    RuntimeGraph graph(model_path);
    const bool init_success = graph.Init();
}