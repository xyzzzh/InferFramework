//
// Created by xyzzzh on 2024/4/1.
//

#ifndef INFERFRAMEWORK_UTILS_HPP
#define INFERFRAMEWORK_UTILS_HPP

#include "Common.hpp"

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
tensor_broadcast(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2);

std::shared_ptr<Tensor>
tensor_padding(const std::shared_ptr<Tensor> &tensor, const std::vector<uint32_t> &pads, float padding_value);

bool tensor_is_same(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2);

std::shared_ptr<Tensor> tensor_add(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2);

void tensor_add(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2,
                const std::shared_ptr<Tensor> &output_tensor);

std::shared_ptr<Tensor> tensor_multiply(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2);

void tensor_multiply(const std::shared_ptr<Tensor> &tensor1, const std::shared_ptr<Tensor> &tensor2,
                     const std::shared_ptr<Tensor> &output_tensor);

std::shared_ptr<Tensor> tensor_create(uint32_t channels, uint32_t rows, uint32_t cols);

std::shared_ptr<Tensor> tensor_create(const std::vector<uint32_t> &shapes);


static std::pair<size_t, size_t> get_mat_size(std::ifstream &file, char split_char){
    bool load_ok = file.good();
    file.clear();
    size_t fn_rows = 0;
    size_t fn_cols = 0;
    const std::ifstream::pos_type start_pos = file.tellg();

    std::string token;
    std::string line_str;
    std::stringstream line_stream;

    while (file.good() && load_ok) {
        std::getline(file, line_str);
        if (line_str.empty()) {
            break;
        }

        line_stream.clear();
        line_stream.str(line_str);
        size_t line_cols = 0;

        std::string row_token;
        while (line_stream.good()) {
            std::getline(line_stream, row_token, split_char);
            ++line_cols;
        }
        if (line_cols > fn_cols) {
            fn_cols = line_cols;
        }

        ++fn_rows;
    }
    file.clear();
    file.seekg(start_pos);
    return {fn_rows, fn_cols};
}

static arma::fmat load_data(const std::string &file_path, char split_char = ','){
    arma::fmat _data;
    if (file_path.empty()) {
        LOG(ERROR) << "CSV file path is empty: " << file_path;
        return _data;
    }

    std::ifstream in(file_path);
    if (!in.is_open() || !in.good()) {
        LOG(ERROR) << "Open file failed: " << file_path;
        return _data;
    }

    std::string line_str;
    std::stringstream line_stream;


    const auto &[rows, cols] = get_mat_size(in, split_char);
    _data.zeros(rows, cols);

    size_t row = 0;
    while (in.good()) {
        std::getline(in, line_str);
        if (line_str.empty()) {
            break;
        }

        std::string token;
        line_stream.clear();
        line_stream.str(line_str);

        size_t col = 0;
        while (line_stream.good()) {
            std::getline(line_stream, token, split_char);
            try {
                _data.at(row, col) = std::stof(token);
            }
            catch (std::exception &e) {
                DLOG(ERROR) << "Parse CSV File meet error: " << e.what() << " row:" << row << " col:" << col;
            }
            col += 1;
            CHECK(col <= cols) << "There are excessive elements on the column";
        }

        row += 1;
        CHECK(row <= rows) << "There are excessive elements on the row";
    }
    return _data;
}


#endif //INFERFRAMEWORK_UTILS_HPP
