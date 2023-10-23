# Selu Implementation In GRU

## Execution Steps
1. Clone The Repository
    ```
    cd pytorch-with-selu
    git checkout CPU
    ```
2. Install Miniconda (if not installed)
    ```
    bash miniconda-installation.sh
    ```
3. Run The Script To Build The PyTorch With Selu (CPU-Only)
    ```
    bash script.sh
    ```

## Approaches
### For CPU
- **Approach** : We have typecast the at::Tensor to torch::Tensor and then applied the selu implemented by the torch itself. And then again converted the torch::Tensor to at::Tensor.

- **Code Modifications** : We have added some lines of code in the file RNN.cpp that can be found @ *pytorch/aten/src/ATen/native/RNN.cpp*
    - Include the torch 
         >  #include <torch/torch.h>
    
    - Add the following function on line#773

        ```
        Tensor applySeLU(Tensor new_gate)
        {
        torch::Tensor input = (torch::Tensor) new_gate;
        torch::Tensor result = torch::selu(input);
        const Tensor result_return = (Tensor) result;
        return result_return;
        }
        ```
    - Modify the struct GRUCell following our function, and modify the last lines of code for new_gate as following.
        -  **Original**
        
            ```
            const auto new_gate =
            chunked_igates[2].add(chunked_hgates[2].mul_(reset_gate)).tanh_();
            return (hidden - new_gate).mul_(input_gate).add_(new_gate);
            ```
        - **Replace With**
        
            ```
            const auto new_gate =
            chunked_igates[2].add(chunked_hgates[2].mul_(reset_gate));
            const auto new_gate_selu = applySeLU(new_gate);
            return (hidden - test_gate).mul_(input_gate).add_(new_gate_selu);
            ```

- Now Build the package again and test the output.

### For GPU (Pending..)
- **Exploration of Approaches**
    1.  We experimented with inserting print statements into the same file used for the CPU build. This was done to gain insights into which section of code is being employed when handling CUDA input. Since there was an 'if' statement to verify if the input is using CUDA, it was crucial to pinpoint the specific code path. However, it was discovered that this file was not being invoked on the GPU.
    
    2.  We identified a CUDA implementation of the GRU (Gated Recurrent Unit) in the aten/src/ATen/native/cuda/RNN.cu file. In this implementation, the GRU cell was designed with a tanh function applied to the new_gate. We created a custom function for the SELU activation and replaced the tanh function with SELU. After testing this modification, we observed that the output remained unchanged, and it was identical to the output produced with the original tanh activation.
