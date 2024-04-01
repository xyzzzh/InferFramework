//
// Created by xyzzzh on 2024/3/29.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/Tensor.hpp"
#include "Utils.hpp"

TEST(test_tensor, tensor_init1) {
    Tensor f1(3, 224, 224);
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, tensor_init2) {
    Tensor f1(std::vector<uint32_t>{3, 224, 224});
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, copy_construct1) {
    Tensor f1(3, 224, 224);
    f1.rand();
    Tensor f2(f1);
    ASSERT_EQ(f2.channels(), 3);
    ASSERT_EQ(f2.rows(), 224);
    ASSERT_EQ(f2.cols(), 224);

    ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, copy_construct2) {

    Tensor f1(3, 2, 1);
    Tensor f2(3, 224, 224);
    f2.rand();
    f1 = f2;
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);

    ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, copy_construct3) {

    Tensor f1(3, 2, 1);
    Tensor f2(std::vector<uint32_t>{3, 224, 224});
    f2.rand();
    f1 = f2;
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);

    ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, move_construct1) {

    Tensor f1(3, 2, 1);
    Tensor f2(3, 224, 224);
    f1 = std::move(f2);
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);

    ASSERT_EQ(f2.data().memptr(), nullptr);
}

TEST(test_tensor, move_construct2) {

    Tensor f2(3, 224, 224);
    Tensor f1(std::move(f2));
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);

    ASSERT_EQ(f2.data().memptr(), nullptr);
}

TEST(test_tensor, set_data) {

    Tensor f2(3, 224, 224);
    arma::fcube cube1(224, 224, 3);
    cube1.randn();
    f2.set_data(cube1);

    ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}

TEST(test_tensor, data) {

    Tensor f2(3, 224, 224);
    f2.fill(1.f);
    arma::fcube cube1(224, 224, 3);
    cube1.fill(1.);
    f2.set_data(cube1);

    ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}

TEST(test_tensor, empty) {

    Tensor f2;
    ASSERT_EQ(f2.empty(), true);

    Tensor f3(3, 3, 3);
    ASSERT_EQ(f3.empty(), false);
}

TEST(test_tensor, transform1) {


    Tensor f3(3, 3, 3);
    ASSERT_EQ(f3.empty(), false);
    f3.transform([](const float &value) { return 1.f; });
    for (int i = 0; i < f3.size(); ++i) {
        ASSERT_EQ(f3.index(i), 1.f);
    }
}

TEST(test_tensor, transform2) {


    Tensor f3(3, 3, 3);
    ASSERT_EQ(f3.empty(), false);
    f3.fill(1.f);
    f3.transform([](const float &value) { return value * 2.f; });
    for (int i = 0; i < f3.size(); ++i) {
        ASSERT_EQ(f3.index(i), 2.f);
    }
}

TEST(test_tensor, clone) {


    Tensor f3(3, 3, 3);
    ASSERT_EQ(f3.empty(), false);
    f3.rand();

    const auto &f4 = f3.clone();
    assert(f4->data().memptr() != f3.data().memptr());
    ASSERT_EQ(f4->size(), f3.size());
    for (int i = 0; i < f3.size(); ++i) {
        ASSERT_EQ(f3.index(i), f4->index(i));
    }
}

TEST(test_tensor, raw_ptr) {


    Tensor f3(3, 3, 3);
    ASSERT_EQ(f3.raw_ptr(), f3.data().mem);
}

TEST(test_tensor, index1) {

    Tensor f3(3, 3, 3);
    ASSERT_EQ(f3.empty(), false);
    std::vector<float> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back(1);
    }
    f3.fill(values);
    for (int i = 0; i < 27; ++i) {
        ASSERT_EQ(f3.index(i), 1);
    }
}

TEST(test_tensor, index2) {

    Tensor f3(3, 3, 3);
    f3.index(3) = 4;
    ASSERT_EQ(f3.index(3), 4);
}

TEST(test_tensor, flatten1) {


    Tensor f3(3, 3, 3);
    std::vector<float> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back(float(i));
    }
    f3.fill(values);
    f3.flatten(false);
    ASSERT_EQ(f3.channels(), 1);
    ASSERT_EQ(f3.rows(), 27);
    ASSERT_EQ(f3.cols(), 1);
    ASSERT_EQ(f3.index(0), 0);
    ASSERT_EQ(f3.index(1), 3);
    ASSERT_EQ(f3.index(2), 6);

    ASSERT_EQ(f3.index(3), 1);
    ASSERT_EQ(f3.index(4), 4);
    ASSERT_EQ(f3.index(5), 7);

    ASSERT_EQ(f3.index(6), 2);
    ASSERT_EQ(f3.index(7), 5);
    ASSERT_EQ(f3.index(8), 8);
}

TEST(test_tensor, flatten2) {


    Tensor f3(3, 3, 3);
    std::vector<float> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back(float(i));
    }
    f3.fill(values);
    f3.flatten(true);
    for (int i = 0; i < 27; ++i) {
        ASSERT_EQ(f3.index(i), i);
    }
}

TEST(test_tensor, fill1) {


    Tensor f3(3, 3, 3);
    std::vector<float> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back(float(i));
    }
    f3.fill(values);
    int index = 0;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < f3.rows(); ++i) {
            for (int j = 0; j < f3.cols(); ++j) {
                ASSERT_EQ(f3.at(c, i, j), index);
                index += 1;
            }
        }
    }
}

TEST(test_tensor, create1) {

    const std::shared_ptr<Tensor> &tensor_ptr = tensor_create(3, 32, 32);
    ASSERT_EQ(tensor_ptr->empty(), false);
    ASSERT_EQ(tensor_ptr->channels(), 3);
    ASSERT_EQ(tensor_ptr->rows(), 32);
    ASSERT_EQ(tensor_ptr->cols(), 32);
}

TEST(test_tensor, create2) {

    const std::shared_ptr<Tensor> &tensor_ptr = tensor_create({3, 32, 32});
    ASSERT_EQ(tensor_ptr->empty(), false);
    ASSERT_EQ(tensor_ptr->channels(), 3);
    ASSERT_EQ(tensor_ptr->rows(), 32);
    ASSERT_EQ(tensor_ptr->cols(), 32);
}

TEST(test_tensor, tensor_broadcast1) {

    const std::shared_ptr<Tensor> &tensor1 = tensor_create({3, 1, 1});
    const std::shared_ptr<Tensor> &tensor2 = tensor_create({3, 32, 32});

    const auto &[tensor11, tensor21] = tensor_broadcast(tensor1, tensor2);
    ASSERT_EQ(tensor21->channels(), 3);
    ASSERT_EQ(tensor21->rows(), 32);
    ASSERT_EQ(tensor21->cols(), 32);

    ASSERT_EQ(tensor11->channels(), 3);
    ASSERT_EQ(tensor11->rows(), 32);
    ASSERT_EQ(tensor11->cols(), 32);

    ASSERT_TRUE(
            arma::approx_equal(tensor21->data(), tensor2->data(), "absdiff", 1e-4));
}

TEST(test_tensor, tensor_broadcast2) {

    const std::shared_ptr<Tensor> &tensor1 = tensor_create({3, 32, 32});
    const std::shared_ptr<Tensor> &tensor2 = tensor_create({3, 1, 1});
    tensor2->rand();

    const auto &[tensor11, tensor21] = tensor_broadcast(tensor1, tensor2);
    ASSERT_EQ(tensor21->channels(), 3);
    ASSERT_EQ(tensor21->rows(), 32);
    ASSERT_EQ(tensor21->cols(), 32);

    ASSERT_EQ(tensor11->channels(), 3);
    ASSERT_EQ(tensor11->rows(), 32);
    ASSERT_EQ(tensor11->cols(), 32);

    for (uint32_t i = 0; i < tensor21->channels(); ++i) {
        float c = tensor2->at(i, 0, 0);
        const auto &in_channel = tensor21->slice(i);
        for (uint32_t j = 0; j < in_channel.size(); ++j) {
            ASSERT_EQ(in_channel.at(j), c);
        }
    }
}

TEST(test_tensor, fill2) {


    Tensor f3(3, 3, 3);
    f3.fill(1.f);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < f3.rows(); ++i) {
            for (int j = 0; j < f3.cols(); ++j) {
                ASSERT_EQ(f3.at(c, i, j), 1.f);
            }
        }
    }
}

TEST(test_tensor, add1) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(1.f);
    const auto &f2 = std::make_shared<Tensor>(3, 224, 224);
    f2->fill(2.f);

    const auto &f3 = tensor_add(f2, f1);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 3.f);
    }
}

TEST(test_tensor, add2) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(1.f);
    const auto &f2 = std::make_shared<Tensor>(3, 1, 1);
    f2->fill(2.f);
    const auto &f3 = tensor_add(f2, f1);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 3.f);
    }
}

TEST(test_tensor, add3) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(1.f);
    const auto &f2 = std::make_shared<Tensor>(3, 1, 1);
    f2->fill(2.f);
    const auto &f3 = tensor_add(f1, f2);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 3.f);
    }
}

TEST(test_tensor, add4) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(1.f);
    const auto &f2 = std::make_shared<Tensor>(3, 224, 224);
    f2->fill(2.f);
    const auto &f3 = tensor_add(f1, f2);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 3.f);
    }
}

TEST(test_tensor, mul1) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(3.f);
    const auto &f2 = std::make_shared<Tensor>(3, 224, 224);
    f2->fill(2.f);
    const auto &f3 = tensor_multiply(f2, f1);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 6.f);
    }
}

TEST(test_tensor, mul2) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(3.f);
    const auto &f2 = std::make_shared<Tensor>(3, 1, 1);
    f2->fill(2.f);
    const auto &f3 = tensor_multiply(f2, f1);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 6.f);
    }
}

TEST(test_tensor, mul3) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(3.f);
    const auto &f2 = std::make_shared<Tensor>(3, 224, 224);
    f2->fill(2.f);
    const auto &f3 = tensor_multiply(f1, f2);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 6.f);
    }
}

TEST(test_tensor, mul4) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(3.f);
    const auto &f2 = std::make_shared<Tensor>(3, 1, 1);
    f2->fill(2.f);
    const auto &f3 = tensor_multiply(f1, f2);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 6.f);
    }
}

TEST(test_tensor, mul5) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(3.f);
    const auto &f2 = std::make_shared<Tensor>(3, 1, 1);
    f2->fill(2.f);

    const auto &f3 = std::make_shared<Tensor>(3, 224, 224);
    tensor_multiply(f1, f2, f3);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 6.f);
    }
}

TEST(test_tensor, mul6) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(3.f);
    const auto &f2 = std::make_shared<Tensor>(3, 1, 1);
    f2->fill(2.f);

    const auto &f3 = std::make_shared<Tensor>(3, 224, 224);
    tensor_multiply(f2, f1, f3);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 6.f);
    }
}

TEST(test_tensor, add5) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(3.f);
    const auto &f2 = std::make_shared<Tensor>(3, 1, 1);
    f2->fill(2.f);

    const auto &f3 = std::make_shared<Tensor>(3, 224, 224);
    tensor_add(f1, f2, f3);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 5.f);
    }
}

TEST(test_tensor, add6) {

    const auto &f1 = std::make_shared<Tensor>(3, 224, 224);
    f1->fill(3.f);
    const auto &f2 = std::make_shared<Tensor>(3, 1, 1);
    f2->fill(2.f);

    const auto &f3 = std::make_shared<Tensor>(3, 224, 224);
    tensor_add(f2, f1, f3);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 5.f);
    }
}

TEST(test_tensor, shapes) {

    Tensor f3(2, 3, 4);
    const std::vector<uint32_t> &shapes = f3.shapes();
    ASSERT_EQ(shapes.at(0), 2);
    ASSERT_EQ(shapes.at(1), 3);
    ASSERT_EQ(shapes.at(2), 4);
}

TEST(test_tensor, raw_shapes1) {

    Tensor f3(2, 3, 4);
    f3.reshape({24});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 1);
    ASSERT_EQ(shapes.at(0), 24);
}

TEST(test_tensor, raw_shapes2) {

    Tensor f3(2, 3, 4);
    f3.reshape({4, 6});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 2);
    ASSERT_EQ(shapes.at(0), 4);
    ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_shapes3) {

    Tensor f3(2, 3, 4);
    f3.reshape({4, 3, 2});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 3);
    ASSERT_EQ(shapes.at(0), 4);
    ASSERT_EQ(shapes.at(1), 3);
    ASSERT_EQ(shapes.at(2), 2);
}

TEST(test_tensor, raw_view1) {

    Tensor f3(2, 3, 4);
    f3.reshape({24}, true);
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 1);
    ASSERT_EQ(shapes.at(0), 24);
}

TEST(test_tensor, raw_view2) {

    Tensor f3(2, 3, 4);
    f3.reshape({4, 6}, true);
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 2);
    ASSERT_EQ(shapes.at(0), 4);
    ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_view3) {

    Tensor f3(2, 3, 4);
    f3.reshape({4, 3, 2}, true);
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 3);
    ASSERT_EQ(shapes.at(0), 4);
    ASSERT_EQ(shapes.at(1), 3);
    ASSERT_EQ(shapes.at(2), 2);
}

TEST(test_tensor, padding1) {

    Tensor tensor(3, 4, 5);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 4);
    ASSERT_EQ(tensor.cols(), 5);

    tensor.fill(1.f);
    tensor.padding({1, 2, 3, 4}, 0);
    ASSERT_EQ(tensor.rows(), 7);
    ASSERT_EQ(tensor.cols(), 12);

    int index = 0;
    for (int c = 0; c < tensor.channels(); ++c) {
        for (int r = 0; r < tensor.rows(); ++r) {
            for (int c_ = 0; c_ < tensor.cols(); ++c_) {
                if ((r >= 2 && r <= 4) && (c_ >= 3 && c_ <= 7)) {
                    ASSERT_EQ(tensor.at(c, r, c_), 1.f) << c << " "
                                                        << " " << r << " " << c_;
                }
                index += 1;
            }
        }
    }
}

TEST(test_tensor, padding2) {

    Tensor tensor(3, 4, 5);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 4);
    ASSERT_EQ(tensor.cols(), 5);

    tensor.fill(1.f);
    tensor.padding({2, 2, 2, 2}, 3.14f);
    ASSERT_EQ(tensor.rows(), 8);
    ASSERT_EQ(tensor.cols(), 9);

    int index = 0;
    for (int c = 0; c < tensor.channels(); ++c) {
        for (int r = 0; r < tensor.rows(); ++r) {
            for (int c_ = 0; c_ < tensor.cols(); ++c_) {
                if (c_ <= 1 || r <= 1) {
                    ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
                } else if (c >= tensor.cols() - 1 || r >= tensor.rows() - 1) {
                    ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
                }
                if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
                    ASSERT_EQ(tensor.at(c, r, c_), 1.f);
                }
                index += 1;
            }
        }
    }
}

TEST(test_tensor, review1) {

    Tensor tensor(3, 4, 5);
    std::vector<float> values;
    for (int i = 0; i < 60; ++i) {
        values.push_back(float(i));
    }

    tensor.fill(values);

    tensor.reshape({4, 3, 5}, true);
    auto data = tensor.slice(0);
    int index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_EQ(data.at(i, j), index);
            index += 1;
        }
    }
    data = tensor.slice(1);
    index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_EQ(data.at(i, j), index + 15);
            index += 1;
        }
    }
    index = 0;
    data = tensor.slice(2);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_EQ(data.at(i, j), index + 30);
            index += 1;
        }
    }

    index = 0;
    data = tensor.slice(3);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_EQ(data.at(i, j), index + 45);
            index += 1;
        }
    }
}

TEST(test_tensor, review2) {

    arma::fmat f1 =
            "1,2,3,4;"
            "5,6,7,8";

    arma::fmat f2 =
            "1,2,3,4;"
            "5,6,7,8";
    std::shared_ptr<Tensor> data = tensor_create(2, 2, 4);
    data->slice(0) = f1;
    data->slice(1) = f2;
    data->reshape({16}, true);
    for (uint32_t i = 0; i < 8; ++i) {
        ASSERT_EQ(data->index(i), i + 1);
    }

    for (uint32_t i = 8; i < 15; ++i) {
        ASSERT_EQ(data->index(i - 8), i - 8 + 1);
    }
}

TEST(test_tensor, review3) {

    arma::fmat f1 =
            "1,2,3,4;"
            "5,6,7,8";

    std::shared_ptr<Tensor> data = tensor_create(1, 2, 4);
    data->slice(0) = f1;
    data->reshape({4, 2}, true);

    arma::fmat data2 = data->slice(0);
    ASSERT_EQ(data2.n_rows, 4);
    ASSERT_EQ(data2.n_cols, 2);
    uint32_t index = 1;
    for (uint32_t row = 0; row < data2.n_rows; ++row) {
        for (uint32_t col = 0; col < data2.n_cols; ++col) {
            ASSERT_EQ(data2.at(row, col), index);
            index += 1;
        }
    }
}

TEST(test_tensor, review4) {

    arma::fmat f1 =
            "1,2,3,4;"
            "5,6,7,8";

    arma::fmat f2 =
            "9,10,11,12;"
            "13,14,15,16";

    std::shared_ptr<Tensor> data = tensor_create(2, 2, 4);
    data->slice(0) = f1;
    data->slice(1) = f2;
    data->reshape({4, 2, 2}, true);
    for (uint32_t c = 0; c < data->channels(); ++c) {
        const auto &in_channel = data->slice(c);
        ASSERT_EQ(in_channel.n_rows, 2);
        ASSERT_EQ(in_channel.n_cols, 2);
        float n1 = in_channel.at(0, 0);
        float n2 = in_channel.at(0, 1);
        float n3 = in_channel.at(1, 0);
        float n4 = in_channel.at(1, 1);
        ASSERT_EQ(n1, c * 4 + 1);
        ASSERT_EQ(n2, c * 4 + 2);
        ASSERT_EQ(n3, c * 4 + 3);
        ASSERT_EQ(n4, c * 4 + 4);
    }
}

TEST(test_tensor, reshape1) {

    arma::fmat f1 =
            "1,3;"
            "2,4";

    arma::fmat f2 =
            "1,3;"
            "2,4";

    std::shared_ptr<Tensor> data = tensor_create(2, 2, 2);
    data->slice(0) = f1;
    data->slice(1) = f2;
    data->reshape({8});
    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(data->index(i), i + 1);
    }

    for (uint32_t i = 4; i < 8; ++i) {
        ASSERT_EQ(data->index(i - 4), i - 4 + 1);
    }
}

TEST(test_tensor, reshape2) {

    arma::fmat f1 =
            "0,2;"
            "1,3";

    arma::fmat f2 =
            "0,2;"
            "1,3";

    std::shared_ptr<Tensor> data = tensor_create(2, 2, 2);
    data->slice(0) = f1;
    data->slice(1) = f2;
    data->reshape({2, 4});
    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(data->index(i), i);
    }

    for (uint32_t i = 4; i < 8; ++i) {
        ASSERT_EQ(data->index(i), i - 4);
    }
}

TEST(test_tensor, ones) {

    Tensor tensor(3, 4, 5);
    tensor.ones();
    for (int i = 0; i < tensor.size(); ++i) {
        ASSERT_EQ(tensor.index(i), 1.f);
    }
}

TEST(test_tensor, rand) {

    Tensor tensor(3, 4, 5);
    tensor.fill(99.f);
    tensor.rand();  // 0 ~ 1
    for (int i = 0; i < tensor.size(); ++i) {
        ASSERT_NE(tensor.index(i), 99.f);
    }
}

TEST(test_tensor, get_data) {

    Tensor tensor(3, 4, 5);
    ASSERT_EQ(tensor.channels(), 3);
    arma::fmat in2(4, 5);
    const arma::fmat &in1 = tensor.slice(0);
    tensor.slice(0) = in2;
    const arma::fmat &in3 = tensor.slice(0);
    ASSERT_EQ(in1.memptr(), in3.memptr());
}

TEST(test_tensor, at1) {

    Tensor tensor(3, 4, 5);
    tensor.at(0, 1, 2) = 2;
    ASSERT_EQ(tensor.at(0, 1, 2), 2);
}

TEST(test_tensor, at2) {

    Tensor tensor(3, 4, 5);
    arma::fmat f(4, 5);
    f.fill(1.f);
    tensor.slice(0) = f;
    ASSERT_TRUE(arma::approx_equal(f, tensor.slice(0), "absdiff", 1e-4));
}

TEST(test_tensor, at3) {

    Tensor tensor(3, 4, 5);
    tensor.fill(1.2f);
    for (uint32_t c = 0; c < tensor.channels(); ++c) {
        for (uint32_t r = 0; r < tensor.rows(); ++r) {
            for (uint32_t c_ = 0; c_ < tensor.cols(); ++c_) {
                ASSERT_EQ(tensor.at(c, r, c_), 1.2f);
            }
        }
    }
}

TEST(test_tensor, is_same1) {

    std::shared_ptr<Tensor> in1 =
            std::make_shared<Tensor>(3, 32, 32);
    in1->fill(2.f);

    std::shared_ptr<Tensor> in2 =
            std::make_shared<Tensor>(3, 32, 32);
    in2->fill(2.f);
    ASSERT_EQ(tensor_is_same(in1, in2), true);
}

TEST(test_tensor, is_same2) {

    std::shared_ptr<Tensor> in1 =
            std::make_shared<Tensor>(3, 32, 32);
    in1->fill(1.f);

    std::shared_ptr<Tensor> in2 =
            std::make_shared<Tensor>(3, 32, 32);
    in2->fill(2.f);
    ASSERT_EQ(tensor_is_same(in1, in2), false);
}

TEST(test_tensor, is_same3) {

    std::shared_ptr<Tensor> in1 =
            std::make_shared<Tensor>(3, 32, 32);
    in1->fill(1.f);

    std::shared_ptr<Tensor> in2 =
            std::make_shared<Tensor>(3, 31, 32);
    in2->fill(1.f);
    ASSERT_EQ(tensor_is_same(in1, in2), false);
}

TEST(test_tensor, tensor_padding1) {

    std::shared_ptr<Tensor> tensor = tensor_create(3, 4, 5);
    ASSERT_EQ(tensor->channels(), 3);
    ASSERT_EQ(tensor->rows(), 4);
    ASSERT_EQ(tensor->cols(), 5);

    tensor->fill(1.f);
    tensor = tensor_padding(tensor, {2, 2, 2, 2}, 3.14f);
    ASSERT_EQ(tensor->rows(), 8);
    ASSERT_EQ(tensor->cols(), 9);

    int index = 0;
    for (int c = 0; c < tensor->channels(); ++c) {
        for (int r = 0; r < tensor->rows(); ++r) {
            for (int c_ = 0; c_ < tensor->cols(); ++c_) {
                if (c_ <= 1 || r <= 1) {
                    ASSERT_EQ(tensor->at(c, r, c_), 3.14f);
                } else if (c >= tensor->cols() - 1 || r >= tensor->rows() - 1) {
                    ASSERT_EQ(tensor->at(c, r, c_), 3.14f);
                }
                if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
                    ASSERT_EQ(tensor->at(c, r, c_), 1.f);
                }
                index += 1;
            }
        }
    }
}