//
// Created by Mason on 2024/10/11.
//

#include <data/Tensor.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor_values, tensor_values1) {
    using namespace BatmanInfer;
    Tensor<float> f1(2, 3, 4);
    f1.Rand();
    f1.Show();

    // 返回第一个通道(channel)中的数据
    LOG(INFO) << "Data in the first channel: " << f1.slice(0);
    // 返回(1, 1, 1)位置的数据
    LOG(INFO) << "Data in the (1, 1, 1): " << f1.at(1, 1, 1);
}

TEST(test_fill_reshape, fill1) {
    // 行主序的填充方式
    using namespace BatmanInfer;
    Tensor<float> f1(2, 3, 4);
    std::vector<float> values(2 * 3 * 4);
    // 将1到24填充到values中
    for (int i = 0; i < 24; ++i)
        values.at(i) = float(i + 1);
    f1.Fill(values);
    f1.Show();
}

float MinusOne(float value) {
    return value - 1.f;
}

float DoubleRet(float value) {
    return value * 2;
}

TEST(test_tranform, transform1) {
    using namespace BatmanInfer;
    Tensor<float> f1(2, 3, 4);
    f1.Rand();
    f1.Show();
    f1.Transform(MinusOne);
    f1.Show();
}

TEST(test_transform, transform2) {
    using namespace BatmanInfer;
    Tensor<float> f2(2, 3, 4);
    f2.Ones();
    f2.Show();
    f2.Transform(DoubleRet);
    f2.Show();
}