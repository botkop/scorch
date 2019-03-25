//
// Created by nazar on 20.12.18.
//

#ifndef TORCH_APP_FOURIERNET_H
#define TORCH_APP_FOURIERNET_H

#include <torch/all.h>

#include <iostream>
#include <vector>



class FourierNet : torch::nn::Module {


public:

    FourierNet(int size) {

        fc1 = torch::nn::Linear(1, size);
        fc2 = torch::nn::Linear(1, 5);
        c1 = torch::randn({size, 1}, torch::requires_grad());
        c2 = torch::randn({5, 1}, torch::requires_grad());
        c = torch::randn({1}, torch::requires_grad());
        sigma = torch::arange(-6, -1, torch::requires_grad());

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        params = parameters(true);
//        params.push_back(fc1->weight);
//        params.push_back(fc1->bias);
//        params.push_back(fc2->weight);
//        params.push_back(fc2->bias);
        params.push_back(c1);
        params.push_back(c2);
        params.push_back(c);
        params.push_back(sigma);
        optimizer = torch::optim::Adam(params, torch::optim::AdamOptions(0.01));
    }


    torch::Tensor forward(torch::Tensor& x) {
        return  (fc2->forward(x).mul(0.1).cos().pow(2).mul(sigma)).exp().matmul(c2 / 10.0f) + fc1->forward(x).mul(0.1).cos().matmul(c1 / 10.0f) + c;
    }


    torch::Tensor weighted_mse( torch::Tensor& y1,  torch::Tensor& y2,  torch::Tensor& weights) {
        torch::Tensor lseq = (y1 - y2).pow(2);
        return  lseq.reshape({1, -1}).matmul(weights);
    }

    std::vector<float> train(std::vector<float>& data, int steps, std::vector<float>& weights) {
        int n = data.size();
        // std::cout << "n=" << n << std::endl;
        torch::Tensor x = torch::arange(0, n).reshape({n, 1});
        torch::Tensor y = torch::tensor(data).reshape({n, 1});
        torch::Tensor ws = torch::tensor(weights).reshape({n, 1});

        torch::Tensor y_pred;
        torch::Tensor loss;

        torch::nn::Module::train();

        for (int t = 0; t < steps; t++) {
            optimizer.zero_grad();

            y_pred = forward(x);

            auto l1_reg = c1.abs().mean() + 0.05 * c2.abs().mean();

            loss = weighted_mse(y_pred, y, ws).reshape(1);
            (loss + l1_reg).backward();
            optimizer.step();

            // std::cout << c1.data<float>()[0] << std::endl;
            // std::cout << loss.template item<float>() << std::endl;
            // println(loss)
        }

        std::vector<float> y_pred_vec = std::vector<float>(y_pred.data<float>(), y_pred.data<float>() + n);

        return y_pred_vec;

    }

private:
    torch::nn::Linear fc1 = nullptr;
    torch::nn::Linear fc2 = nullptr;
    torch::Tensor c1;
    torch::Tensor c2;
    torch::Tensor c;
    torch::Tensor sigma;

    torch::optim::Adam optimizer = torch::optim::Adam(parameters(), torch::optim::AdamOptions(0.01));
    std::vector<torch::Tensor> params;


};


#endif //TORCH_APP_FOURIERNET_H
