//
// Created by Mason on 2024/10/16.
//
#include "runtime/runtime_ir.hpp"
#include "status_code.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace BatmanInfer {
    RuntimeGraph::RuntimeGraph(std::string model_path) : model_path_(std::move(model_path)) {}


    const std::string &RuntimeGraph::model_path() {
        return this->model_path_;
    }

    void RuntimeGraph::set_model_path(const std::string& model_path) {
        this->model_path_ = model_path;
    }

    bool RuntimeGraph::Init() {
        if (this->model_path_.empty()) {
            LOG(ERROR) << "The model path is empty";
            return false;
        }

        this->graph_ = std::make_unique<ONNXGraph>();
        int load_result = this->graph_->load(model_path_);
        if (load_result != 0) {
            LOG(ERROR) << "Can not find the model path: " << model_path_;
            return false;
        }

        std::vector<ONNXOperator *> operators = this->graph_->operators;
        if (operators.empty()) {
            LOG(ERROR) << "Can not read the layers' define";
            return false;
        }

        this->operators_.clear();
        this->operators_maps_.clear();
        for (const ONNXOperator *op: operators) {
            if (!op) {
                LOG(ERROR) << "Meet the empty node";
                continue;
            } else {
                std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
                // 初始化算子的名称
                runtime_operator->name = op->name;
                runtime_operator->type = op->type;

                // 初始化算子中的input
                const std::vector<ONNXOperand *> &inputs = op->inputs;
                if (!inputs.empty()) {
                    InitGraphOperatorsInput(inputs, runtime_operator);
                }

                // 记录输出operand中的名称
                const std::vector<ONNXOperand *> &outputs = op->outputs;
                if (!outputs.empty()) {
                    InitGraphOperatorsOutput(outputs, runtime_operator);
                }

                // 初始化算子中的attribute(权重)
                const std::map<std::string, ONNXAttribute> &attrs = op->attrs;
                if (!attrs.empty())
                    InitGraphAttrs(attrs, runtime_operator);
            }
        }
        return true;
    }

    void RuntimeGraph::InitGraphOperatorsInput(const std::vector<ONNXOperand *> &inputs,
                                               const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const ONNXOperand *input: inputs) {
            if (!input) {
                continue;
            }
            const ONNXOperator *producer = input->producer;
            std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
            runtime_operand->name = producer->name;
            runtime_operand->shapes = input->shape;

            switch (input->type) {
                case 1:
                    runtime_operand->type = RuntimeDataType::kTypeFloat32;
                    break;
                case 0:
                    runtime_operand->type = RuntimeDataType::kTypeUnknown;
                    break;
                default:
                    LOG(FATAL) << "Unknown input operand type: " << input->type;
            }
            runtime_operator->input_operands.insert({producer->name, runtime_operand});
            runtime_operator->input_operands_seq.push_back(runtime_operand);
        }
    }

    void RuntimeGraph::InitGraphOperatorsOutput(const std::vector<ONNXOperand *> &outputs,
                                                const std::shared_ptr<RuntimeOperator> &runtime_operator) {
         for (const ONNXOperand *output: outputs) {
             if (!output)
                 continue;
             const auto& consumers = output->consumers;
             for (const auto& c : consumers)
                 runtime_operator->output_names.push_back(c->name);
         }
    }

    void RuntimeGraph::InitGraphAttrs(const std::map<std::string, ONNXAttribute> &attrs,
                                      const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const auto &[name, attr]: attrs) {
            switch (attr.type) {
                case 1: {
                    std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>();
                    runtime_attribute->type = RuntimeDataType::kTypeFloat32;
                    runtime_attribute->weight_data = attr.data;
                    runtime_attribute->shape = attr.shape;
                    runtime_operator->attribute.insert({name, runtime_attribute});
                    break;
                }
                default: {
                    LOG(FATAL) << "Unknown attribute type: " << attr.type;
                }
            }
        }
    }


}