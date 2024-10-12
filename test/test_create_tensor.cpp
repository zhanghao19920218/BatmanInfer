//
// Created by Mason on 2024/10/11.
//

#include <data/Tensor.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor, tensor_init1D) {
    using namespace BatmanInfer;
    Tensor<float> f1(4);
    f1.Fill(1.f);
    f1.Show();
}
