//
// Created by Mason on 2024/10/15.
//

#ifndef BATMAN_INFER_RUNTIME_OP_HPP
#define BATMAN_INFER_RUNTIME_OP_HPP

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "runtime/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace BatmanInfer {
    class Layer;

    /**
     * 计算图中的计算节点
     */
    struct RuntimeOperator {
        virtual ~RuntimeOperator();

        bool has_forward = false;
        // 计算节点的名称
        std::string name;
        // 计算节点的类型
        std::string type;
        // 节点对应的计算Layer
        std::shared_ptr<Layer> layer;
        // 节点的输出节点名称
        std::vector<std::string> output_names;
        // 节点的输出操作数
        std::vector<RuntimeOperand> output_operands;

        // 节点的输入操作数
        std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;
        // 节点的输入操作数，顺序排列
        std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;
        // 输出节点的名字和节点对应
        std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;

        // 算子的参数信息
        std::map<std::string, RuntimeParameter*> params;
        // 算子的属性信息, 内涵权重信息
        std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
    };

    class RuntimeOperatorUtils {
    public:
        static void InitOperatorInput(
                const std::vector<std::shared_ptr<RuntimeOperator>>& operators);
    };
}

#endif //BATMAN_INFER_RUNTIME_OP_HPP
