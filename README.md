# Shield-MPPI: Safety-Enhanced Model Predictive Path Integral Control

This repository provides a GPU-accelerated implementation of the **Shield Model Predictive Path Integral (Shield-MPPI)** algorithm using **PyTorch**. Shield-MPPI integrates Control Barrier Functions (CBFs) into the MPPI framework, enhancing safety and robustness in trajectory optimization.


## 🔗 Reference:
> - Yin, J., Dawson, C., Fan, C., & Tsiotras, P. (2023).  
>   *Shield model predictive path integral: A computationally efficient robust MPC method using control barrier functions*.  
>   **IEEE Robotics and Automation Letters, 8(11), 7106-7113.**  
>   [IEEE Xplore Link](https://ieeexplore.ieee.org/abstract/document/10250917)

## 🛠 Base Code Reference

This implementation builds upon the base MPPI structure provided by the following project:  
[https://github.com/MizuhoAOKI/python_simple_mppi](https://github.com/MizuhoAOKI/python_simple_mppi)

---

## ⚙️ Prerequisites

- Python 3.10+
- PyTorch with GPU support
- CUDA 11.8+
- Recommended: Use a virtual environment (e.g., conda)

### ✅ Create Virtual Environment

```bash
conda create -n python310 python=3.10
conda activate python310
conda install pytorch=2.5.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Installation And Run
```bash
git clone <repository_url>
cd <repository_name>/main
python3 shield_mppi_main.py
```
---

## 🎬 Experiment Results

### Performance Comparison: Standard MPPI vs Shield-MPPI

| Method        | k = 20                             | k = 1000                              |
|---------------|------------------------------------|---------------------------------------|
| Standard MPPI | ![](gifs/standard_k20.gif)         | ![](gifs/standard_k1000.gif)          |
| Shield-MPPI   | ![](gifs/shield_k20.gif)           | ![](gifs/shield_k1000.gif)            |


---


