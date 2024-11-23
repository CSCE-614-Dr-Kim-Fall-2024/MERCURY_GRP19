# CSCE 614 Fall 2024 Project: **Architectural Study of MERCURY: Accelerating DNN Training by Exploiting Input Similarity**

This document provides instructions on running the modified PyTorch implementation designed for the MERCURY architecture. The implementation introduces functionality to analyze and exploit input similarity for accelerating deep neural network (DNN) training.

## Running the Modified PyTorch Implementation

To ensure smooth execution and isolate dependencies, it is recommended to create a **virtual environment** before proceeding.

### Step 1: Clone the Repository and Navigate to the `pytorch` Directory

```bash
git clone https://github.com/CSCE-614-Dr-Kim-Fall-2024/MERCURY_GRP19.git
cd MERCURY_GRP19/pytorch
```

### Step 2: Install the Modified PyTorch

Install the customized version of PyTorch by running the following command within the `pytorch` directory:

```bash
pip install -e .
```

> **Note**: The `-e` flag ensures the package is installed in "editable" mode, allowing you to modify the source code and immediately reflect the changes without reinstallation.

### Step 3: Run the Provided Script

After successfully installing the modified PyTorch, execute the `resnet_val.py` script to evaluate the ResNet model using the MERCURY-based implementation.

```bash
python3 resnet_val.py
```

### Step 4: Observe the Output

The execution logs will be stored in the following location:

```
results/results_cached/out_resnet/resnet_run.log
```

### Understanding the Hitmap File

For detailed analysis, the system generates a file named **`hitmap-n.csv`**, where `n` corresponds to the layer number of the ResNet model. This file provides insights into the input similarity exploitation mechanism for the `n`th layer.

#### Hitmap File Format:

| **Input Shape (4 values)** | **Signature Shape (2 values)** | **Hit Count** | **Miss Count** | **Hitmap (Same Length as Signature)** |
|-----------------------------|-------------------------------|---------------|----------------|---------------------------------------|

- **Input Shape**: Describes the dimensions of the input tensor processed by the layer (e.g., batch size, channels, height, width).
- **Signature Shape**: Represents the reduced representation of inputs based on their similarity, using MERCURY's optimization techniques.
- **Hit Count**: The number of inputs that matched the signature in this layer, indicating reuse opportunities.
- **Miss Count**: The number of inputs that failed to match, requiring full computation.
- **Hitmap**: A detailed record showing which elements matched the signature.

By analyzing the **hitmap file**, you can assess how effectively MERCURY's input similarity optimizations improve computational efficiency at each layer of the model.
