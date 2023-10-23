# Selu Implementation In GRU

## Execution Steps
1. Clone The Repository
    ```
    cd pytorch-with-selu
    git checkout GPU
    ```
2. Install Miniconda (if not installed)
    ```
    bash miniconda-installation.sh
    ```
3. Modify the CUDA Version
    ```
    - Replace the 110 on line 15 in script.sh with your CUDA version (110 represents CUDA version 11.0)
    ```
4. Run The Script To Build The PyTorch With Selu 

    ```
    bash script.sh
    ```

## Approaches
### For GPU
- **Exploration of Approaches**
    1.  We experimented with inserting print statements into the same file used for the CPU build. This was done to gain insights into which section of code is being employed when handling CUDA input. Since there was an 'if' statement to verify if the input is using CUDA, it was crucial to pinpoint the specific code path. However, it was discovered that this file was not being invoked on the GPU.
    
    2.  We identified a CUDA implementation of the GRU (Gated Recurrent Unit) in the aten/src/ATen/native/cuda/RNN.cu file. In this implementation, the GRU cell was designed with a tanh function applied to the new_gate. We created a custom function for the SELU activation and replaced the tanh function with SELU.
    
    - Issue 

        After testing this modification, we observed that the output remained unchanged, and it was identical to the output produced with the original tanh activation.
