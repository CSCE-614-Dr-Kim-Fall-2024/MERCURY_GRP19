# CSCE 614 Fall 2024 Project: **Architectural Study of MERCURY: Accelerating DNN Training by Exploiting Input Similarity**

## Group Members
- Karthikeyan Renga Rajan (535004635)

- Kartikeya Aryan Agarwal (236002811)

- Neel Shah (534002778)

- Anirudh Suresh Bharadwaj (535005834)

## Project Description
This project implements and evaluates the MERCURY technique, which accelerates deep neural network (DNN) training by exploiting input similarity. We focus only on forward propagation and convolutional layers, using PyTorch for caching implementation and SCALE-Sim for hardware simulation of computation cycles. Our study covers AlexNet, ResNet101, and GoogleNet models, simulated on an Eyeriss-like hardware configuration with a weight-stationary dataflow.

## Running the Modified PyTorch Implementation

Below are the steps/instructions for running the modified PyTorch implementation designed for the MERCURY technique. The implementation introduces functionality to analyze and exploit input similarity for accelerating DNN training.

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

After successfully installing the modified PyTorch, execute the `resnet_val.py` script to evaluate the ResNet model using the MERCURY-based implementation:

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

## Running the Modified SCALE-Sim Implementation

### SCALE-Sim Inputs

SCALE-Sim uses two input files to run, a configuration file and a topology file, as follows: 

#### Configuration file

The configuration file is used to specify the architecture and run parameters for the simulations.
The following shows a sample config file:

![sample config](https://github.com/scalesim-project/scale-sim-v2/blob/main/documentation/resources/config-file-example.png "sample config")

The config file has three sections. The "*general*" section specifies the run name, which is user specific. The "*architecture_presets*" section describes the parameter of the systolic array hardware to simulate.
The "*run_preset*" section specifies if the simulator should run with user specified bandwidth, or should it calculate the optimal bandwidth for stall free execution.

#### Topology file

The topology file is a *CSV* file which decribes the layers of the workload topology. The layers are typically described as convolution layer parameters as shown in the example below.

![sample topo](https://github.com/scalesim-project/scale-sim-v2/blob/main/documentation/resources/topo-file-example.png "sample topo")

Below are the steps/instructions for running the modified SCALE-Sim implementation designed for accounting the reduced computations due the MERCURY caching technique. The implementation introduces functionality to integrate the hitmap data generated from our PyTorch implementation into SCALE-Sim, allowing us to skip redundant computations.

### Step 1: Navigate to the `scalesim` Directory and Install it

Within the cloned MERCURY_GRP19 repository, go inside the 'scalesim' directory and install the customized version of SCALE-Sim:

```bash
cd MERCURY_GRP19/scalesim/
pip install -e .
```

### Step 2: Enter the Main `scalesim` Directory

Navigate to the main 'scalesim' directory by running the following command within the directory `MERCURY_GRP19/scalesim/`: 

```bash
cd scalesim/
```

### Step 3: Run the Provided Script

Execute the SCALE-Sim script with the following command:

```bash
python3 scale.py -c <path_to_config_file> -t <path_to_topology_file> -p <path_to_output_log_dir> -m <path_to_results_cached_directory>
```

Replace the placeholders with appropriate file paths:
- <path_to_config_file>: Path to the configuration file (e.g., `MERCURY_GRP19/scalesim/configs/mercury.cfg`)
- <path_to_topology_file>: Path to the topology file (e.g., `MERCURY_GRP19/scalesim/topologies/mercury/{model}.csv`)
- <path_to_output_log_dir>: Directory where output you want the logs to be stored (e.g., `MERCURY_GRP19/scalesim/output/Baseline (or SigVec or Cache)/{model}/`)
- <path_to_results_cached_directory>: Path to the results_cached directory (e.g., `MERCURY_GRP19/results/results_cached2/out_{model}`)

> **Note**: The **`-m` flag and the corresponding path** is only required for running simulations with caching, and use **{model}_SigVec.csv** topology files to determine signature vector computation cycles on the same hardware configuration (mercury.cfg). Also, replace {model} with any of the AlexNet, Googlenet or ResNet101.

### Step 4: Observe the Output

After running the script, navigate to the output directory to view the results:

```bash
cd ..
cd output/Baseline/{model}/output.log
```

Similarly, navigate to any of the Baseline, SigVec, or Cache directory inside the output directory to see the results and logs. These contain computation cycles among other metrics for baseline model, for signature vector calculation on the same hardware configuration (Eyeriss-like), or for cached model inferencing, respectively.

### The Summary Logs

The simulator also generates read write traces and summary logs at `MERCURY_GRP19/scalesim/output/Baseline/{model}/mercury/`. There are three summary logs:

- **COMPUTE_REPORT.csv**: Layer wise logs for **compute cycles**, stalls, utilization percentages etc.

- **BANDWIDTH_REPORT.csv**: Layer wise information about average and maximum bandwidths for each operand when accessing SRAM and DRAM

- **DETAILED_ACCESS_REPORT.csv**: Layer wise information about number of accesses and access cycles for each operand for SRAM and DRAM

## References
- Janfaza, V., Weston, K., Razavi, M., Mandal, S., Mahmud, F., Hilty, A. and Muzahid, A., 2023, February. Mercury: Accelerating dnn training by exploiting input similarity. In 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA) (pp. 638-650). IEEE.

- Paszke, A. et al., 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32. Curran Associates, Inc., pp. 8024–8035.

- Samajdar, A., Joseph, J.M., Zhu, Y., Whatmough, P., Mattina, M. and Krishna, T., 2020, August. A systematic methodology for characterizing scalability of dnn accelerators using scale-sim. In 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS) (pp. 58-68). IEEE.

- Samajdar, A., Zhu, Y., Whatmough, P., Mattina, M. and Krishna, T., 2018. Scale-sim: Systolic cnn accelerator simulator. arXiv preprint arXiv:1811.02883.
