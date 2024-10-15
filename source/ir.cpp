//
// Created by Mason on 2024/10/14.
//
#include <runtime/ir.h>
#include <onnx_conv/OnnxUtils.hpp>
#include <google/protobuf/util/json_util.h>
#include <onnx/onnx.pb.h>


#include <climits>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>
#include <utility>

namespace BatmanInfer {
    static bool type_is_integer(int type) {
        if (type == 1) return false;
        if (type == 2) return false;
        if (type == 3) return false;
        if (type == 4) return true;
        if (type == 5) return true;
        if (type == 6) return true;
        if (type == 7) return true;
        if (type == 8) return true;
        if (type == 9) return true;
        if (type == 10) return false;
        if (type == 11) return false;
        if (type == 12) return false;
        return false;
    }

    static const char *type_to_string(int type) {
        if (type == 1) return "f32";
        if (type == 2) return "f64";
        if (type == 3) return "f16";
        if (type == 4) return "i32";
        if (type == 5) return "i64";
        if (type == 6) return "i16";
        if (type == 7) return "i8";
        if (type == 8) return "u8";
        if (type == 9) return "bool";
        if (type == 10) return "cp64";
        if (type == 11) return "cp128";
        if (type == 12) return "cp32";
        return "null";
    }

    static const char *type_to_numpy_string(int type) {
        if (type == 1) return "float32";
        if (type == 2) return "float64";
        if (type == 3) return "float16";
        if (type == 4) return "int32";
        if (type == 5) return "int64";
        if (type == 6) return "int16";
        if (type == 7) return "int8";
        if (type == 8) return "uint8";
        if (type == 9) return "bool8";
        if (type == 10) return "csingle";
        if (type == 11) return "cdouble";
        if (type == 12) return "chalf";
        return "null";
    }

    static const char *type_to_dtype_string(int type) {
        if (type == 1) return "torch.float";
        if (type == 2) return "torch.double";
        if (type == 3) return "torch.half";
        if (type == 4) return "torch.int";
        if (type == 5) return "torch.long";
        if (type == 6) return "torch.short";
        if (type == 7) return "torch.int8";
        if (type == 8) return "torch.uint8";
        if (type == 9) return "torch.bool";
        if (type == 10) return "torch.complex64";
        if (type == 11) return "torch.complex128";
        if (type == 12) return "torch.complex32";
        return "null";
    }

    static size_t type_to_elemsize(int type) {
        if (type == 1) return 4;
        if (type == 2) return 8;
        if (type == 3) return 2;
        if (type == 4) return 4;
        if (type == 5) return 8;
        if (type == 6) return 2;
        if (type == 7) return 1;
        if (type == 8) return 1;
        if (type == 9) return 1;
        if (type == 10) return 8;
        if (type == 11) return 16;
        if (type == 12) return 4;
        return 0; // null
    }

    static int string_to_type(const char *s) {
        if (strcmp(s, "f32") == 0) return 1;
        if (strcmp(s, "f64") == 0) return 2;
        if (strcmp(s, "f16") == 0) return 3;
        if (strcmp(s, "i32") == 0) return 4;
        if (strcmp(s, "i64") == 0) return 5;
        if (strcmp(s, "i16") == 0) return 6;
        if (strcmp(s, "i8") == 0) return 7;
        if (strcmp(s, "u8") == 0) return 8;
        if (strcmp(s, "bool") == 0) return 9;
        if (strcmp(s, "cp64") == 0) return 10;
        if (strcmp(s, "cp128") == 0) return 11;
        if (strcmp(s, "cp32") == 0) return 12;
        return 0; // null
    }

    bool operator==(const ONNXParameter &lhs, const ONNXParameter &rhs) {
        if (lhs.type != rhs.type)
            return false;

        if (lhs.type == 0)
            return true;

        if (lhs.type == 1 && lhs.b == rhs.b)
            return true;

        if (lhs.type == 2 && lhs.i == rhs.i)
            return true;

        if (lhs.type == 3 && lhs.f == rhs.f)
            return true;

        if (lhs.type == 4 && lhs.s == rhs.s)
            return true;

        if (lhs.type == 5 && lhs.ai == rhs.ai)
            return true;

        if (lhs.type == 6 && lhs.af == rhs.af)
            return true;

        if (lhs.type == 7 && lhs.as == rhs.as)
            return true;

        return false;
    }

    ONNXAttribute::ONNXAttribute(const onnx::TensorProto &tensor) {
        // 初始化 Shape
        for (int i = 0; i < tensor.dims_size(); ++i)
            shape.emplace_back(tensor.dims(i));

        // 检查是否有原始值(现在只有float32类型的)
        if (tensor.has_raw_data()) {
            const std::string& raw_data = tensor.raw_data();
            data.resize(raw_data.size() / sizeof(float));
            std::memcpy(data.data(), raw_data.data(), raw_data.size());
        } else {
            // 使用 float_data 字段
            data.assign(tensor.float_data().begin(), tensor.float_data().end());
        }

        // 确定数据类型并初始化数据
        switch (tensor.data_type()) {
            case onnx::TensorProto::FLOAT:
                type = 1; // f32
                break;

            case onnx::TensorProto::DOUBLE:
                type = 2; // f64
                break;

            case onnx::TensorProto::INT32:
                type = 4; // i32
                break;

            case onnx::TensorProto::INT64:
                type = 5; // i64
                break;

                // 处理其他数据类型
            default:
//                // 处理 raw_data
//                if (!tensor.raw_data().empty()) {
//                    data.resize(tensor.raw_data().size());
//                    memcpy(data.data(), tensor.raw_data().data(), data.size());
//                    // 根据 raw_data 和 shape 设置正确的 type
//                    // 这里可以根据实际需求设置 type
//                }
                break;
        }
    }

    bool operator==(const ONNXAttribute &lhs, const ONNXAttribute &rhs) {
        if (lhs.type != rhs.type)
            return false;

        if (lhs.type == 0)
            return true;

        if (lhs.shape != rhs.shape)
            return false;

        if (lhs.data != rhs.data)
            return false;

        return true;
    }

    ONNXAttribute operator+(const ONNXAttribute &a, const ONNXAttribute &b) {
        ONNXAttribute c;

        if (a.type != b.type) {
            std::cerr << "concat attribute type mismatch\n";
            return c;
        }

        if (a.shape.size() != b.shape.size()) {
            std::cerr << "concat attribute shape rank mismatch\n";
            return c;
        }

        for (int i = 1; i < static_cast<int>(a.shape.size()); i++) {
            if (a.shape[i] != b.shape[i]) {
                std::cerr << "concat attribute shape mismatch\n";
                return c;
            }
        }

        c.type = a.type;
        c.shape = a.shape;
        c.shape[0] += b.shape[0]; // concat the first dim

        c.data.resize(a.data.size() + b.data.size());
        memcpy(c.data.data(), a.data.data(), a.data.size());
        memcpy(c.data.data() + a.data.size(), b.data.data(), b.data.size());

        return c;
    }

    ONNXParameter ONNXParameter::parse_from_string(const std::string &value) {
        ONNXParameter p;
        p.type = 0;

        if (value == "None" || value == "()" || value == "[]") {
            return p;
        }

        if (value == "True" || value == "False") {
            // bool
            p.type = 1;
            p.b = value == "True";
            return p;
        }

        if (value[0] == '(' || value[0] == '[') {
            // list
            std::string lc = value.substr(1, value.size() - 2);
            std::istringstream lcss(lc);

            while (!lcss.eof()) {
                std::string elem;
                std::getline(lcss, elem, ',');

                if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) ||
                    (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9'))) {
                    // string
                    p.type = 7;
                    p.as.push_back(elem);
                } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
                    // float
                    p.type = 6;
                    p.af.push_back(std::stof(elem));
                } else {
                    // integer
                    p.type = 5;
                    p.ai.push_back(std::stoi(elem));
                }
            }
            return p;
        }

        if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) ||
            (value[0] == '-' && (value[1] < '0' || value[1] > '9'))) {
            // string
            p.type = 4;
            p.s = value;
            return p;
        }

        if (value.find('.') != std::string::npos || value.find('e') != std::string::npos) {
            // float
            p.type = 3;
            p.f = std::stof(value);
            return p;
        }

        // integer
        p.type = 2;
        p.i = std::stoi(value);
        return p;
    }

    ONNXGraph::ONNXGraph() = default;

    ONNXGraph::~ONNXGraph() {
        for (auto x: operators)
            delete x;

    }

    ONNXGraph::ONNXGraph(const BatmanInfer::ONNXGraph &rhs) {}

    ONNXGraph &ONNXGraph::operator=(const BatmanInfer::ONNXGraph &rhs) {
        return *this;
    }

    /**
     * 是不是权重的输入
     * @param input_name
     * @param graph
     * @return
     */
    bool is_initializer(const std::string &input_name, const onnx::GraphProto &graph) {
        for (const auto &initializer: graph.initializer()) {
            if (initializer.name() == input_name) {
                return true; // Input is an initializer (weight)
            }
        }
        return false;
    }

    static void load_parameter(ONNXOperator *op,
                               const std::string &key,
                               const std::string &value) {
        op->params[key] = ONNXParameter::parse_from_string(value);
    }

    /**
     * 打印Tensor信息
     * @param tensor
     */
    static void print_tensor_info(const onnx::ValueInfoProto& tensor, ONNXOperand* operand)
    {
        operand->name = tensor.name();
        const auto& type = tensor.type().tensor_type();
        int data_type = type.elem_type();
        operand->type = map_onnx_type_to_custom_type(data_type);
        operand->shape.clear();
        for (const auto& dim : type.shape().dim()) {
            operand->shape.push_back(dim.dim_value());
        }

        std::cout << "Tensor name: " << operand->name << "\t";
        std::cout << "Data type: " << operand->type << "\t";
        std::cout << "Shape: ";
        for (const auto& dim : operand->shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    // 查找输出张量的信息
    void find_output_tensor_info(const std::string& output_name,
                                 const onnx::GraphProto& graph,
                                 ONNXOperator *op) {
        for (const auto& value_info : graph.value_info()) {
            if (value_info.name() == output_name) {
                auto operand = new ONNXOperand();
                print_tensor_info(value_info, operand);
                op->outputs.emplace_back(operand);
                return;
            }
        }
    }

    void find_input_tensor_info(const std::string& input_name,
                                const onnx::GraphProto& graph,
                                ONNXOperator *op)
    {
        for (const auto& node : graph.node()) {
            for (const auto& output_name : node.output()) {
                if (output_name == input_name) {
                    std::cout << "Found input from node: " << node.op_type() << std::endl;
                    for (const auto& input : node.input()) {
                        if (!is_initializer(input, graph)) {
                            find_input_tensor_info(input, graph, op);
                        }
                    }
                    return;
                }
            }
        }

        // 如果在节点中找不到，检查图的输入
        for (const auto& input_tensor : graph.input()) {
            if (input_tensor.name() == input_name) {
                std::cout << "Input tensor is an input to the graph." << std::endl;
                auto operand = new ONNXOperand();
                print_tensor_info(input_tensor, operand);
                op->input_names.emplace_back(input_name);
                op->inputs.emplace_back(operand);
                return;
            }
        }
    }


    static void load_input_key(ONNXOperator *op,
                               const onnx::NodeProto& node,
                               const onnx::GraphProto& graph) {
        op->inputs.clear();
        op->input_names.clear();
        for (const auto& input_name: node.input()) {
            // 查找不是权重的值
            if (!is_initializer(input_name, graph)) {
                std::cout << "Input name: " << input_name << "\n";
                find_input_tensor_info(input_name, graph, op);
            }
        }

        for (const auto& output_name : node.output()) {
            // 查找不是权重的值
            if (!is_initializer(output_name, graph)) {
                std::cout << "Output name: " << output_name << "\n";
                find_output_tensor_info(output_name, graph, op);
            }
        }
    }

    /**
     * 从Node结点里获取ONNXAttribute
     * @param node
     * @param attribute_name
     * @return
     */
    ONNXAttribute get_attribute_from_node(const onnx::TensorProto &tensor) {
        ONNXAttribute custom_attr;

        custom_attr = ONNXAttribute(tensor);
        return custom_attr;
    }

    /**
     * 查看是否是权重
     * @param input_name 输入的参数名
     * @param graph
     * @return
     */
    const onnx::TensorProto *is_initializer_tensor(const std::string &input_name, const onnx::GraphProto &graph) {
        for (const auto &initializer: graph.initializer()) {
            if (initializer.name() == input_name) {
                return &initializer; // Input is an initializer (weight)
            }
        }
        return nullptr;
    }



    /**
     * Function to get the weight names from a node
     * @param node 节点信息
     * @return
     */
    std::map<std::string, const onnx::TensorProto *> get_weight_names_from_node(const onnx::NodeProto &node,
                                                                                const onnx::GraphProto &graph) {
        std::map<std::string, const onnx::TensorProto *> weight_names;

        // Iterate through the node's inputs
        for (int i = 0; i < node.input_size(); ++i) {
            const std::string &input_name = node.input(i);

            // Check if this input is part of the initializer (i.e., it's a weight)
            auto tensor = is_initializer_tensor(input_name, graph);
            if (tensor) {
                weight_names[input_name] = tensor;
            }
        }

        return weight_names;
    }

    /**
     * 解析配置文件或参数设置
     * @param op 自定义操作符
     * @param node 操作符结点
     * @param graph 图的结构
     */
    void load_attribute(ONNXOperator *op,
                        const onnx::NodeProto &node,
                        const onnx::GraphProto &graph) {
        auto attribute_names = get_weight_names_from_node(node, graph);
        for (const auto &attr_info: attribute_names) {
            ONNXAttribute a = get_attribute_from_node(*attr_info.second);
            op->attrs[attr_info.first] = a;
        }
    }

    void ONNXGraph::load(const std::string &model_path) {

        // 读取ONNX模型文件
        onnx::ModelProto modelProto;

        // read ONNX Model
        bool success = onnx_read_proto_from_binary(model_path.c_str(),
                                                   reinterpret_cast<google::protobuf::Message *>(&modelProto));

        // 读取 ONNX 模型文件
        if (!success) {
            fprintf(stderr, "Failed to read ONNX model from %s\n", model_path.c_str());
            return;
        }

        // 读取操作符和操作数的数量
        int operator_count = 0;
        int operand_count = 0;

        getOperatorAndOperandCount(modelProto,
                                   operator_count,
                                   operand_count);

        // 获取算子图
        auto graph = modelProto.graph();

        // 获取算子的信息
        for (int i = 0; i < operator_count; ++i) {
            // 获取每一个算子
            const onnx::NodeProto &node = graph.node(i);

            std::string type = node.op_type();
            std::string name = node.name();
            int input_count = node.input_size();
            int output_count = node.output_size();

            ONNXOperator *op = new_operator(type, name);

            for (int j = 0; j < input_count; j++) {
                // 获取第 j 个输入的名称
                const std::string &operand_name = node.input(j);

                // 获取输入的操作数
                ONNXOperand *r = new_operand(operand_name);
                r->consumers.push_back(op);
                // 输入的消费者
                op->inputs.push_back(r);
            }

            for (int j = 0; j < output_count; j++) {
                std::string operand_name;

                ONNXOperand *r = new_operand(operand_name);
                r->producer = op;
                op->outputs.emplace_back(r);
            }

            // 对操作符进行权重参数加载
            load_attribute(op, node, graph);

            // 对操作数进行权重参数加载
            load_input_key(op, node, graph);
        }

    }

    ONNXOperand *ONNXGraph::get_operand(const std::string &name) {
        for (ONNXOperand *r: operands) {
            if (r->name == name)
                return r;
        }
        return nullptr;
    }

    ONNXOperator *ONNXGraph::new_operator(const std::string &type, const std::string &name) {
        auto *op = new ONNXOperator;
        op->type = type;
        op->name = name;
        operators.emplace_back(op);
        return op;
    }

    ONNXOperand *ONNXGraph::new_operand(const std::string &name) {
        auto *r = new ONNXOperand;
        r->name = name;
        operands.emplace_back(r);
        return r;
    }

    ONNXOperator *
    ONNXGraph::new_operator_before(const std::string &type, const std::string &name, const ONNXOperator *cur) {
        auto *op = new ONNXOperator;
        op->type = type;
        op->name = name;
        operators.insert(std::find(operators.begin(), operators.end(), cur), op);
        return op;
    }

    ONNXOperator *
    ONNXGraph::new_operator_after(const std::string &type, const std::string &name, const ONNXOperator *cur) {
        auto *op = new ONNXOperator;
        op->type = type;
        op->name = name;
        operators.insert(std::find(operators.begin(), operators.end(), cur) + 1, op);
        return op;
    }

    const ONNXOperand *ONNXGraph::get_operand(const std::string &name) const {
        for (const ONNXOperand *r : operands)
        {
            if (r->name == name)
                return r;
        }
        return nullptr;
    }

}