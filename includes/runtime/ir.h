//
// Created by Mason on 2024/10/13.
//

#ifndef BATMANINFER_IR_H
#define BATMANINFER_IR_H

#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <onnx/onnx_pb.h>
#include <onnx/proto_utils.h>

namespace BatmanInfer {
    class ONNXParameter {
    public:
        ONNXParameter() : type(0) {}

        ONNXParameter(bool _b) : type(1), b(_b) {}

        ONNXParameter(int _i) : type(2), i(_i) {}

        ONNXParameter(long _l) : type(2), i(_l) {}

        ONNXParameter(long long _l) : type(2), i(_l) {}

        ONNXParameter(float _f) : type(3), f(_f) {}

        ONNXParameter(double _f) : type(3), f(_f) {}

        ONNXParameter(const char *_s) : type(4), s(_s) {}

        ONNXParameter(const std::string &_s) : type(4), s(_s) {}

        ONNXParameter(const std::initializer_list<int> &_ai) : type(5), ai(_ai) {}

        ONNXParameter(const std::initializer_list<int64_t> &_ai) : type(5) {
            for (const auto &x: _ai)
                ai.push_back((int) x);
        }

        ONNXParameter(const std::initializer_list<float> &_af)
                : type(6), af(_af) {
        }

        ONNXParameter(const std::initializer_list<double> &_af)
                : type(6) {
            for (const auto &x: _af)
                af.push_back((float) x);
        }

        ONNXParameter(const std::vector<float> &_af)
                : type(6), af(_af) {
        }

        ONNXParameter(const std::initializer_list<const char *> &_as)
                : type(7) {
            for (const auto &x: _as)
                as.push_back(std::string(x));
        }

        ONNXParameter(const std::initializer_list<std::string> &_as)
                : type(7), as(_as) {
        }

        ONNXParameter(const std::vector<std::string> &_as)
                : type(7), as(_as) {
        }

        static ONNXParameter parse_from_string(const std::string& value);


        // 0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
        int type;

        // value
        bool b;
        int i;
        float f;
        std::vector<int> ai;
        std::vector<float> af;

        // keep std::string typed member the last for cross cxxabi compatibility
        std::string s;
        std::vector<std::string> as;
    };

    class ONNXAttribute {
    public:
        ONNXAttribute() : type(0) {}

        ONNXAttribute(const std::string& name, const onnx::TensorProto& tensor)
                : name(name), tensor(tensor), type(1) {}

        std::string name;
        onnx::TensorProto tensor;
        int type;
    };

    class ONNXOperand {
    public:
        std::string name;
        std::vector<ONNXParameter> params;
        std::vector<int> shape;
    };

    class ONNXOperator {
    public:
        std::string name;
        std::string type;
        std::vector<ONNXOperand> inputs;
        std::vector<ONNXOperand> outputs;
        std::map<std::string, ONNXAttribute> attributes;
    };

    class ONNXGraph {
    public:
        ONNXGraph() {}

        void load(const std::string& model_path) {
            onnx::ModelProto model;
//            onnx::LoadProtoFromPath(model_path, &model);
            // Load model and populate operators and operands
        }

        void save(const std::string& model_path) {
            onnx::ModelProto model;
            // Populate model from operators and operands
//            onnx::SaveProtoToPath(model, model_path);
        }

        std::vector<ONNXOperator> operators;
        std::vector<ONNXOperand> operands;
    };
}

#endif //BATMANINFER_IR_H
