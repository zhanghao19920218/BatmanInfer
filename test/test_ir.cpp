//
// Created by Mason on 2024/10/16.
//

#include <data/Tensor.hpp>
#include "runtime/ir.h"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <others/utils.hpp>
#include <string>

TEST(test_new_ir, onnx_graph_ops) {
    using namespace BatmanInfer;
    const std::string& modelPath = "./model/model.onnx";

    // 初始化Protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::unique_ptr<ONNXGraph> graph = std::make_unique<ONNXGraph>();
    int load_result = graph->load(modelPath);
    // 这里加载失败，请首先考虑相对路径错误
    ASSERT_EQ(load_result, 0);
    const auto &ops = graph->operators;
    for (int i = 0; i < ops.size(); ++i)
        LOG(INFO) << ops.at(i)->name;
    // 清理 protobuf
    google::protobuf::ShutdownProtobufLibrary();
}

TEST(test_new_ir, onnx_graph_operands) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/model.onnx";
    std::unique_ptr<ONNXGraph> graph = std::make_unique<ONNXGraph>();
    int load_result = graph->load(model_path);
    // 如果这里加载失败，请首先考虑
    ASSERT_EQ(load_result, 0);
    const auto &ops = graph->operators;
    for (auto op : ops) {
        LOG(INFO) << "OP Name: " << op->name;
        LOG(INFO) << "OP Inputs";
        for (auto & input : op->inputs)
            LOG(INFO) << "Input name: " << input->name
                      << " shape: " << ShapeStr(input->shape);
        LOG(INFO) << "OP Outputs";
        for (auto &output: op->outputs)
            LOG(INFO) << "Output name: " << output->name
                      << " shape: " << ShapeStr(output->shape);
        LOG(INFO) << "---------------------------------------------";
    }
}

// 输出运算数和参数
TEST(test_new_ir, onnx_graph_operands_and_params) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/model.onnx";
    std::unique_ptr<ONNXGraph> graph = std::make_unique<ONNXGraph>();
    int load_result = graph->load(model_path);
    // 如果这里加载失败，请首先考虑
    ASSERT_EQ(load_result, 0);
    const auto &ops = graph->operators;
    for (auto op : ops) {
        if (op->name != "/linear/Gemm")
            continue;
        LOG(INFO) << "OP Name: " << op->name;
        LOG(INFO) << "OP Inputs";
        for (auto & input : op->inputs)
            LOG(INFO) << "Input name: " << input->name
                      << " shape: " << ShapeStr(input->shape);
        LOG(INFO) << "OP Outputs";
        for (auto &output: op->outputs)
            LOG(INFO) << "Output name: " << output->name
                      << " shape: " << ShapeStr(output->shape);

        LOG(INFO) << "Weight: ";
        for (const auto &weight: op->attrs)
            LOG(INFO) << weight.first << " : " << ShapeStr(weight.second.shape)
                      << " type " << weight.second.type;
        LOG(INFO) << "---------------------------------------------";
    }
}

TEST(test_new_ir, onnx_graph_operands_customer_producer) {
    using namespace BatmanInfer;
    const std::string& model_path = "./model/model.onnx";
    std::unique_ptr<ONNXGraph> graph = std::make_unique<ONNXGraph>();
    int load_result = graph->load(model_path);
    // 如果这里加载失败，请首先考虑
    ASSERT_EQ(load_result, 0);
    const auto &operands = graph->operands;
    for (auto operand : operands) {
        LOG(INFO) << "Operand Name: $" << operand->name;
        LOG(INFO) << "Customers: ";
        for (const auto &customer: operand->consumers)
            LOG(INFO) << customer->name;
        LOG(INFO) << "Producer: " << operand->producer->name;
    }

}