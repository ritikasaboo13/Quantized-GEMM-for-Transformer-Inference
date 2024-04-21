#pragma once
#include "modules/linear.cuh"
#include "ops/op_elemwise.cuh"
template <typename T>
class MLP
{
private:
    std::vector<LinearLayer<T>> layers;
    std::vector<int> layer_dims;
    std::vector<Tensor<T>> activ;
    std::vector<Tensor<T>> d_activ;

    int batch_size;
    int in_dim;

public:
    MLP(int batch_size_, int in_dim_, std::vector<int> layer_dims_, bool gpu)
        : batch_size(batch_size_), in_dim(in_dim_), layer_dims(layer_dims_)
    {
        for (int i = 0; i < layer_dims.size(); i++)
        {
            if (i == 0)
            {
                layers.emplace_back(in_dim, layer_dims[i], gpu);
            }
            else
            {
                layers.emplace_back(layer_dims[i - 1], layer_dims[i], gpu);
            }
        }
        // make all the activation tensors
        activ.reserve(layer_dims.size() - 1);
        d_activ.reserve(layer_dims.size() - 1);
        for (int i = 0; i < layer_dims.size() - 1; i++)
        {
            activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
            // technically, i do not need to save d_activ for backprop, but since iterative
            // training does repeated backprops, reserving space for these tensor once is a good idea
            d_activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
        }
    }

    std::vector<Parameter<T> *> parameters()
    {
        std::vector<Parameter<T> *> params;
        for (int i = 0; i < layer_dims.size(); i++)
        {
            auto y = layers[i].parameters();
            params.insert(params.end(), y.begin(), y.end());
        }
        return params;
    }

    void init() {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].init_uniform();
            // layers[i].init_constant();
        }
    }

    //This function peforms the forward operation of a MLP model
    //Specifically, it should call the forward oepration of each linear layer 
    //Except for the last layer, it should invoke Relu activation after each layer.
    void forward(const Tensor<T> &in, Tensor<T> &out)
    {
        // Lab-2: add your code here
        // perform forward pass starting with in and store final result in out
        int index_last_layer = layer_dims.size()-1;
        Tensor<T> z_;
        for (int i = 0; i < layer_dims.size()-1; i++){
            if (i == 0){
                layers[i].forward(in, activ[i]);
            }
            else{
                layers[i].forward(z_, activ[i]);
            }
            op_relu(activ[i], activ[i]);
            z_ = activ[i];
            // std::cout << "Layer " << i+1 << " forward pass done z" << i+1 << "="  << activ[i].shape() << std::endl;
        }
        layers[index_last_layer].forward(activ[index_last_layer - 1], out);
        // std::cout << "Layer " << index_last_layer+1 << " forward pass done z" << index_last_layer+1 << "="  << out.shape() << std::endl;
    }


    //This function perofmrs the backward operation of a MLP model.
    //Tensor "in" is the gradients for the outputs of the last linear layer (aka d_logits from op_cross_entropy_loss)
    //Invoke the backward function of each linear layer and Relu from the last one to the first one.
    void backward(const Tensor<T> &in, const Tensor<T> &d_out, Tensor<T> &d_in)
    {
       //Lab-2: add your code here
       int index_last_layer = layer_dims.size() - 1;
        for (int i = index_last_layer; i >= 0; i--){
            if (i == index_last_layer){
                layers[i].backward(activ[i-1], d_out, d_activ[i-1]);
            }
            else if (i == 0){
                Tensor <T> z_temp{batch_size, layer_dims[i], in.on_device};
                Tensor <T> grad_temp{batch_size, layer_dims[i], in.on_device};
                layers[i].forward(in, z_temp);
                op_relu_back(z_temp, d_activ[i], grad_temp);
                layers[i].backward(in, grad_temp, d_in);
            }
            else{
                Tensor <T> z_temp{batch_size, layer_dims[i], in.on_device};
                Tensor <T> grad_temp{batch_size, layer_dims[i], in.on_device};
                layers[i].forward(activ[i-1], z_temp);
                op_relu_back(z_temp, d_activ[i], grad_temp);
                layers[i].backward(activ[i-1], grad_temp, d_activ[i-1]);
            }
       }
    }
};
