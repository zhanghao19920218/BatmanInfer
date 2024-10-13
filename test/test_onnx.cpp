//
// Created by Mason on 2024/10/13.
//

#include <iostream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fstream>
#include <onnx/onnx_pb.h>
#include <google/protobuf/text_format.h>
#include <others/utils.hpp>

TEST(test_ir, onnx_graph_ops) {
    const std::string& modelPath = "./model/model.onnx";

    // 初始化Protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // 读取ONNX模型文件
    onnx::ModelProto modelProto;
    std::ifstream fin(modelPath, std::ios::in | std::ios::binary);
    if (!fin) {
        LOG(ERROR) <<  "Cannot open file: " << modelPath;
        return;
    }

    if (!modelProto.ParseFromIstream(&fin)) {
        LOG(ERROR) <<  "Failed to parse ONNX model.";
        return;
    }

    fin.close();

    // 获取计算图 (Graph)
    const onnx::GraphProto& graph = modelProto.graph();

    // 获取算子列表 (节点列表)
    const auto& nodes = graph.node();

    // 遍历算子列表，打印每个算子的名称
    for (int i = 0; i < nodes.size(); ++i) {
        const onnx::NodeProto& node = nodes.Get(i);
        // 打印算子名称
        LOG(INFO) << "Operator " << i << ": " << node.op_type();
    }

    // 清理 protobuf
    google::protobuf::ShutdownProtobufLibrary();
}

TEST(test_ir, onnx_graph_operands) {
    using namespace BatmanInfer;
    const std::string& modelPath = "./model/model.onnx";

    // 初始化Protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // 读取ONNX模型文件
    onnx::ModelProto modelProto;
    std::ifstream fin(modelPath, std::ios::in | std::ios::binary);
    if (!fin) {
        LOG(ERROR) <<  "Cannot open file: " << modelPath;
        return;
    }

    if (!modelProto.ParseFromIstream(&fin)) {
        LOG(ERROR) <<  "Failed to parse ONNX model.";
        return;
    }

    fin.close();

    // 获取计算图 (Graph)
    const onnx::GraphProto& graph = modelProto.graph();

    // 构建变量名到 TypeProto 的映射，用于获取形状信息
    std::unordered_map<std::string, onnx::TypeProto> value_types;

    // 处理图的输入
    for (int i = 0; i < graph.input_size(); ++i) {
        const auto& input = graph.input(i);
        value_types[input.name()] = input.type();
    }

    // 处理图的输出
    for (int i = 0; i < graph.output_size(); ++i) {
        const auto& output = graph.output(i);
        value_types[output.name()] = output.type();
    }

    // 处理图的中间的变量
    for (int i = 0; i < graph.value_info_size(); ++i) {
        const auto& value_info = graph.value_info(i);
        value_types[value_info.name()] = value_info.type();
    }

    // 处理初始化变量（权重），构建TypeProto
    for (int i = 0; i < graph.initializer_size(); ++i) {
        const auto& initializer = graph.initializer(i);
        // 创建 TypeProto
        onnx::TypeProto type_proto;
        auto* tensor_type = type_proto.mutable_tensor_type();
        tensor_type->set_elem_type(initializer.data_type());

        // 设置形状
        auto* shape = tensor_type->mutable_shape();
        for (int j = 0; j < initializer.dims_size(); ++j) {
            auto* dim = shape->add_dim();
            dim->set_dim_value(initializer.dims(j));
        }

        value_types[initializer.name()] = type_proto;
    }

    // 现在可以遍历节点（算子）打印信息
    const auto& nodes = graph.node();
    for (int i = 0; i < nodes.size(); ++i) {
        const onnx::NodeProto& node = nodes.Get(i);
        LOG(INFO) << "OP Name: " << node.op_type();

        // 打印输入信息
        LOG(INFO) << "Op Inputs: ";
        for (int j = 0; j < node.input_size(); ++j) {
            const std::string& input_name = node.input(j);
            std::string shape_str = "Unknown";
            if (value_types.find(input_name) != value_types.end()) {
                shape_str = ShapeStr(value_types[input_name]);
            }
            LOG(INFO) << "Input name: " << input_name << " shape: " << shape_str;
        }

        // 打印输出信息
        LOG(INFO) << "OP Outputs";
        for (int j = 0; j < node.output_size(); ++j) {
            const std::string& output_name = node.output(j);
            std::string shape_str = "Unknown";
            if (value_types.find(output_name) != value_types.end()) {
                shape_str = ShapeStr(value_types[output_name]);
            }
            LOG(INFO) << "Output name: " << output_name << " shape: " << shape_str;
        }
    }

    // 清理 protobuf
    google::protobuf::ShutdownProtobufLibrary();

}

TEST(test_ir, onnx_graph_operands_and_params) {
    using namespace BatmanInfer;
    const std::string& modelPath = "./model/model.onnx";

    // 初始化Protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // 读取ONNX模型文件
    onnx::ModelProto modelProto;
    std::ifstream fin(modelPath, std::ios::in | std::ios::binary);
    if (!fin) {
        LOG(ERROR) <<  "Cannot open file: " << modelPath;
        return;
    }

    if (!modelProto.ParseFromIstream(&fin)) {
        LOG(ERROR) <<  "Failed to parse ONNX model.";
        return;
    }

    fin.close();

    // 获取计算图 (Graph)
    const onnx::GraphProto& graph = modelProto.graph();

    // 获取算子列表 (节点列表)
    const auto& nodes = graph.node();

    // 获取初始化器列表 (权重参数)
    std::unordered_map<std::string, onnx::TensorProto> initializerMap;
    for (const auto& initializer : graph.initializer())
        initializerMap[initializer.name()] = initializer;

    // 遍历算子列表，打印每个算子的名称
    for (int i = 0; i < nodes.size(); ++i) {
        const onnx::NodeProto& node = nodes.Get(i);
        // 打印算子名称
        LOG(INFO) << "Operator " << i << ": " << node.op_type();

        // 检查节点的输入
        for (const auto& input : node.input()) {
            if (initializerMap.find(input) != initializerMap.end()) {
                const onnx::TensorProto& weight = initializerMap[input];
                LOG(INFO) << " Weight parameters: " << input ;
            }
        }
    }
}