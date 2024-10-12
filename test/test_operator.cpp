//
// Created by Mason on 2024/10/11.
//
#include <gtest/gtest.h>
#include <armadillo>

// Demonstrate some basic assertions
TEST(HelloTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality
    EXPECT_EQ(7 * 6, 42);
}

TEST(test_arma, add) {
    using namespace arma;
    fmat in_matrix1 = "1, 2, 3;"
                      "4, 5, 6;"
                      "7, 8, 9;";

    fmat in_matrix2 = "1, 2, 3;"
                      "4, 5, 6;"
                      "7, 8, 9;";

    const fmat &out_matrix1 = "2, 4, 6;"
                              "8, 10, 12;"
                              "14, 16, 18;";

    const fmat &out_matrix2 = in_matrix1 + in_matrix2;
    ASSERT_EQ(approx_equal(out_matrix2, out_matrix1, "absdiff", 1e-5), true);
}

TEST(test_arma, sub) {
    using namespace arma;
    fmat in_matrix1 = "1, 2, 3;"
                      "4, 5, 6;"
                      "7, 8, 9;";

    fmat in_matrix2 = "1, 2, 3;"
                      "4, 5, 6;"
                      "7, 8, 9;";

    const fmat &out_matrix1 = "0, 0, 0;"
                              "0, 0, 0;"
                              "0, 0, 0;";

    const fmat &out_matrix2 = in_matrix1 - in_matrix2;
    ASSERT_EQ(approx_equal(out_matrix1, out_matrix2, "absdiff", 1e-5), true);
}

TEST(test_arma, matmul) {
    using namespace arma;
    fmat in_matrix1 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";

    fmat in_matrix2 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";

    const fmat &out_matrix1 = "30,36,42;"
                              "66,81,96;"
                              "102,126,150;";

    const fmat &out_matrix2 = in_matrix1 * in_matrix2;
    ASSERT_EQ(approx_equal(out_matrix1, out_matrix2, "absdiff", 1e-5), true);
}

TEST(test_arma, pointwise) {
    using namespace arma;
    fmat in_matrix1 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";

    fmat in_matrix2 = "1,2,3;"
                      "4,5,6;"
                      "7,8,9";

    const fmat &out_matrix1 = "1,4,9;"
                              "16,25,36;"
                              "49,64,81;";

    const fmat &out_matrix2 = in_matrix1 % in_matrix2;
    ASSERT_EQ(approx_equal(out_matrix1, out_matrix2, "absdiff", 1e-5), true);
}

TEST(test_arma, weights) {
    using namespace arma;
    fmat weights_w = "3, 2, 1";

    fmat in_matrix = "1; 1; 1";

    fmat b_vec = "1";

    const fmat &out_matrix1 = "7";

    const fmat &out_matrix2 = weights_w * in_matrix + b_vec;
    ASSERT_EQ(approx_equal(out_matrix1, out_matrix2, "absdiff", 1e-5), true);
}

TEST(test_arma, e_expr) {
    using namespace arma;

    fmat in_matrix = "0, 1";

    arma::fmat Y = arma::exp(-in_matrix);

    std::cout << "Y = " << Y << std::endl;
}
