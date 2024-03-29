//
// Created by xyzzzh on 2024/3/29.
//

#ifndef INFERFRAMEWORK_CSVDATALOADER_HPP
#define INFERFRAMEWORK_CSVDATALOADER_HPP

#include <armadillo>
#include <string>

class CSVDataLoader {
public:
    // 从csv文件中初始化张量
    static arma::fmat load_data(const std::string &file_path, char split_char = ',');

    // 得到csv文件的尺寸大小，LoadData中根据这里返回的尺寸大小初始化返回的fmat
    static std::pair<size_t, size_t> get_mat_size(std::ifstream &file, char split_char);

};


#endif //INFERFRAMEWORK_CSVDATALOADER_HPP
