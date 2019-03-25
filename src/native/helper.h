//
// Created by nazar on 25.03.19.
//

#ifndef JAVA_TORCH_LIB_HELPER_H
#define JAVA_TORCH_LIB_HELPER_H


#include <torch/all.h>
#include "models/FourierNet.h"


namespace at {

    std::map<int, at::TensorOptions> index2Options = {
            {0, kByte},
            {1, kChar},
            {2, kShort},
            {3, kInt},
            {4, kLong},
            {5, kHalf},
            {6, kFloat},
            {7, kDouble},
            {8, kComplexHalf},
            {9, kComplexFloat},
            {10, kComplexDouble}
    };

    at::TensorOptions create_options(int k) {
        return index2Options[k];
    }

    Tensor concat(Tensor & t1, Tensor & t2, int dim) {
        return at::cat(TensorList({t1, t2}), dim);
    }

    Tensor concat(std::vector<Tensor> ts, int dim) {
        return at::cat(TensorList(ts.data(), ts.size()), dim);
    }

    Tensor maximum(Tensor t, long dim, bool keepdim) {
        return std::get<0>( at::max(t, dim, keepdim) );
    }

    IntList* int_list(size_t size, int* data) {
        long array[size];
        for (int i = 0; i < size; ++i) {
            array[i] = static_cast<long>(data[i]);
        }
        return new IntList(array, size);
    }


//    Tensor make_ones(int dtype, std::vector<long long int> dims) {
//        long array[dims.size()];
//        for (int i = 0; i < dims.size(); ++i) {
//            array[i] = static_cast<long>(dims[i]);
//        }
//        auto t = at::ones(IntList(array, static_cast<size_t>(dims.size())), index2Options[dtype]);
//        return t;
//    }

    std::vector<float> train(std::vector<float>& data, int steps, std::vector<float>& weights) {
        return FourierNet(30).train(data, steps, weights);
    }

}

#endif //JAVA_TORCH_LIB_HELPER_H

