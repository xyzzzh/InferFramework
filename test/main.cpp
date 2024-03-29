//
// Created by xyzzzh on 2024/3/29.
//

#include <iostream>
#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("InferFramework");
    FLAGS_log_dir = "./log/";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start test...\n";
    return RUN_ALL_TESTS();
}