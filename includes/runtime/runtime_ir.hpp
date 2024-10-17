//
// Created by Mason on 2024/10/16.
//

#ifndef BATMAN_INFER_RUNTIME_IR_HPP
#define BATMAN_INFER_RUNTIME_IR_HPP

#include "ir.h"
#include <runtime/runtime_operand.hpp>
#include "runtime_op.hpp"
#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace BatmanInfer {
    /**
     * 计算图结构, 由多个计算节点和节点之间的数据流程图组成
     */
    class RuntimeGraph {
    public:
        /**
         * 初始化计算图
         * @param model_path 模型的路径
         */
        RuntimeGraph(std::string  model_path);

        /**
         * 设置模型路径
         * @param model_path 设置模型路径
         */
        void set_model_path(const std::string& model_path);

        /**
         * 返回模型路径
         * @return
         */
        const std::string &model_path();

        /**
         * 计算图的初始化
         * @return 是否初始化成功
         */
        bool Init();

        /**
         * 获取操作符
         * @return
         */
        const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

    private:
        /**
         * 初始化Batman Infer计算图节点中的输入操作数
         * @param inputs ONNX中的输入操作数
         * @param runtime_operator 计算图节点
         */
        static void InitGraphOperatorsInput(
                const std::vector<ONNXOperand *> & inputs,
                const std::shared_ptr<RuntimeOperator> &runtime_operator);


        /**
         * 初始化
         * @param outputs
         * @param runtime_operator
         */
        static void InitGraphOperatorsOutput(
                const std::vector<ONNXOperand *> &outputs,
                const std::shared_ptr<RuntimeOperator> &runtime_operator);


        /**
         * 初始化Batman Infer 计算图中的节点属性
         * @param attrs ONNX节点属性
         * @param runtime_operator  计算图节点
         */
        static void InitGraphAttrs(const std::map<std::string, ONNXAttribute> &attrs,
                                   const std::shared_ptr<RuntimeOperator> &runtime_operator);

    public:
    private:
        /**
         * 计算图输入节点的名称
         */
        std::string input_name_;

        /**
         * 计算图输出节点的名称
         */
        std::string output_name_;

        /**
         * 模型的文件路径
         */
        std::string model_path_;

        /**
         * 算子
         */
        std::vector<std::shared_ptr<RuntimeOperator>> operators_;
        std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_maps_;

        /**
         * Onnx的graph
         */
        std::unique_ptr<ONNXGraph> graph_;
    };
}

#endif //BATMAN_INFER_RUNTIME_IR_HPP
