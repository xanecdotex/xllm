<!-- Copyright 2022 JD Co.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

[English](./README.md) | [中文](./README_zh.md)

<div align="center">
<img src="docs/assets/logo_with_llm.png" alt="xLLM" style="width:50%; height:auto;">
    
[![Document](https://img.shields.io/badge/Document-black?logo=html5&labelColor=grey&color=red)](https://xllm.readthedocs.io/zh-cn/latest/) [![Docker](https://img.shields.io/badge/Docker-black?logo=docker&labelColor=grey&color=%231E90FF)](https://hub.docker.com/r/xllm/xllm-ai) [![License](https://img.shields.io/badge/license-Apache%202.0-brightgreen?labelColor=grey)](https://opensource.org/licenses/Apache-2.0) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/jd-opensource/xllm) 
    
</div>

---------------------

<p align="center">
| <a href="https://xllm.readthedocs.io/zh-cn/latest/"><b>Documentation</b></a> | 
</p>

## 1. Project Overview

**xLLM** is an **efficient LLM inference framework**, specifically optimized for **Chinese AI accelerators**, enabling enterprise-grade deployment with enhanced efficiency and reduced cost. The framework adopts a **service-engine decoupled** inference architecture, achieving breakthrough efficiency through several  technologies: at the service layer, including elastic scheduling of online/offline requests, dynamic PD disaggregation, a hybrid EPD mechanism for multimodal and high-availability fault tolerance; and at the engine layer, combined with technologies such as multi-stream parallel computing, graph fusion optimization, speculative inference, dynamic load balancing and global KV cache management. The overall architecture is shown below:

<div align="center">
<img src="docs/assets/xllm_arch.png" alt="xllm_arch" style="width:80%; height:auto;">
</div>

**xLLM** already supports efficient deployment of mainstream large models (such as *DeepSeek-V3.1*, *Qwen2/3*, etc.) on Chinese AI accelerators, empowering enterprises to implement high-performance, low-cost AI large model applications. xLLM has been fully deployed in JD.com’s real core retail businesses, covering a variety of scenarios including intelligent customer service, risk control, supply chain optimization, ad recommendation, and more.


## 2. Core Features

**xLLM** delivers robust intelligent computing capabilities. By leveraging hardware system optimization and algorithm-driven decision control, it jointly accelerates the inference process, enabling high-throughput, low-latency distributed inference services.

**Full Graph Pipeline Execution Orchestration**
- Asynchronous decoupled scheduling at the requests scheduling layer, to reduce computational bubbles.
- Asynchronous parallelism of computation and communication at the model graph layer, overlapping computation and communication.
- Pipelining of heterogeneous computing units at the operator kernel layer, overlapping computation and memory access.

**Graph Optimization for Dynamic Shapes**
- Dynamic shape adaptation based on parameterization and multi-graph caching methods to enhance the flexibility of static graph.
- Controlled tensor memory pool to ensure address security and reusability.
- Integration and adaptation of performance-critical custom operators (e.g., *PageAttention*, *AllReduce*).

**Efficient Memory Optimization**
- Mapping management between discrete physical memory and continuous virtual memory.
- On-demand memory allocation to reduce memory fragmentation.
- Intelligent scheduling of memory pages to increase memory reusability.
- Adaptation of corresponding operators for domestic accelerators.

**Global KV Cache Management**
- Intelligent offloading and prefetching of KV in hierarchical caches.
- KV cache-centric distributed storage architecture.
- Intelligent KV routing among computing nodes.

**Algorithm-driven Acceleration**
- Speculative decoding optimization to improve efficiency through multi-core parallelism.
- Dynamic load balancing of MoE experts to achieve efficient adjustment of expert distribution.

---

## 3. Code Architecture
```
├── xllm/
|   : main source folder
│   ├── api_service/               # code for api services
│   ├── core/  
│   │   : xllm core features folder
│   │   ├── common/                
│   │   ├── distributed_runtime/   # code for distributed and pd serving
│   │   ├── framework/             # code for execution orchestration
│   │   ├── kernels/               # adaption for npu kernels adaption
│   │   ├── layers/                # model layers impl
│   │   ├── runtime/               # code for worker and executor
│   │   ├── scheduler/             # code for batch and pd scheduler
│   │   └── util/
│   ├── models/                    # models impl
│   ├── processors/                # code for vlm pre-processing
│   ├── proto/                     # communication protocol
|   └── server/                    # xLLM server
├── examples/                      # examples of calling xLLM
├── tools/                         # code for npu time generations
└── xllm.cpp                       # entrypoint of xLLM
```

Supported models list:
- DeepSeek-V3/R1
- DeepSeek-R1-Distill-Qwen
- Kimi-k2
- Llama2/3
- MiniCPM-V
- MiMo-VL
- Qwen2/2.5/QwQ
- Qwen2.5-VL
- Qwen3 / Qwen3-MoE


---

## 4. Quick Start
#### Installation
First, download the image we provide:
```bash
# A2 x86
docker pull xllm/xllm-ai:xllm-0.6.0-dev-hb-rc2-py3.11-oe24.03-lts
# A2 arm
docker pull xllm/xllm-ai:xllm-0.6.0-dev-hb-rc2-py3.11-oe24.03-lts-aarch64
# A3 arm
docker pull xllm/xllm-ai:xllm-0.6.0-dev-hc-rc2-py3.11-oe24.03-lts-aarch64
# or
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-0.6.0-dev-hb-rc2-py3.11-oe24.03-lts
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-0.6.0-dev-hb-rc2-py3.11-oe24.03-lts-aarch64
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-0.6.0-dev-hc-rc2-py3.11-oe24.03-lts-aarch64
```
Then create the corresponding container:
```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host  --device=/dev/davinci0  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/queue_schedule:/var/queue_schedule -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /export/home:/export/home -w /export/home -v ~/.ssh:/root/.ssh  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /home/:/home/  -v /runtime/:/runtime/ xllm/xllm-ai:xllm-0.6.0-dev-hb-rc2-py3.11-oe24.03-lts
```

Install official repo and submodules：
```bash
git clone https://github.com/jd-opensource/xllm
cd xllm 
git submodule init
git submodule update
```
The compilation depends on [vcpkg](https://github.com/microsoft/vcpkg). The Docker image already includes VCPKG_ROOT preconfigured. If you want to manually set it up, you can:
```bash
git clone https://gitcode.com/xLLM-AI/vcpkg.git
cd vcpkg && git checkout ffc42e97c866ce9692f5c441394832b86548422c
export VCPKG_ROOT=/your/path/to/vcpkg
```
Install python dependencies:
```bash
cd xllm
pip install -r cibuild/requirements-dev.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade setuptools wheel
```

#### Compilation
When compiling, generate executable files `build/xllm/core/server/xllm` under `build/`:
```bash
python setup.py build
```
Or, compile directly using the following command to generate the whl package under `dist/`:
```bash
python setup.py bdist_wheel
```

#### Launch
Run the following command to start xLLM engine: 
```bash
./build/xllm/core/server/xllm \    # launch xllm server
    --model=/path/to/your/llm  \   # model path（to replace with your own path）
    --backend=llm \                # indicate the LLM backend
    --port=9977 \                  # set service port to 9977
    --max_memory_utilization 0.90  # set the maximal utilization of device memory
```

--- 

## 5. Contributing
There are several ways you can contribute to xLLM:

1. Reporting Issues (Bugs & Errors)
2. Suggesting Enhancements
3. Improving Documentation
    + Fork the repository
    + Add your view in document
    + Send your pull request
4. Writing Code
    + Fork the repository
    + Create a new branch
    + Add your feature or improvement
    + Send your pull request

We appreciate all kinds of contributions! 🎉🎉🎉
If you have problems about development, please check our document: **[Document](https://xllm.readthedocs.io/zh-cn/latest)**

---

## 6. Community & Support
If you encounter any issues along the way, you are welcomed to submit reproducible steps and log snippets in the project's Issues area, or contact the xLLM Core team directly via your internal Slack. Moreover, we have established a WeChat user group. You can find our group chat QR code image [here](https://qr.link/JZaROS) or visit the following live QR code. Welcome to contact us!

<div align="center">
  <img src="docs/assets/wechat_qrcode1.png" alt="qrcode1" width="30%" />
  <img src="docs/assets/wechat_qrcode2.png" alt="qrcode2" width="30%" />
</div>

---

## 7. Acknowledgment

This project was made possible thanks to the following open-source projects:  
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM) - xLLM draws inspiration from ScaleLLM's graph construction method and references its runtime execution. 
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Build xLLM hybrid KV cache management based on Mooncake.
- [brpc](https://github.com/apache/brpc) - Build high-performance http service based on brpc.
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) - Build C++ tokenizer based on tokenizers-cpp.
- [safetensors](https://github.com/huggingface/safetensors) - xLLM relies on the C binding safetensors capability.
- [Partial JSON Parser](https://github.com/promplate/partial-json-parser) - Implement xLLM's C++ JSON parser with insights from Python and Go implementations.
- [concurrentqueue](https://github.com/cameron314/concurrentqueue) - A fast multi-producer, multi-consumer lock-free concurrent queue for C++11.


Thanks to the following collaborating university laboratories:

- [THU-MIG](https://ise.thss.tsinghua.edu.cn/mig/projects.html) (School of Software, BNRist, Tsinghua University)
- USTC-Cloudlab (Cloud Computing Lab, University of Science and Technology of China)
- [Beihang-HiPO](https://github.com/buaa-hipo) (Beihang HiPO research group)
- PKU-DS-LAB (Data Structure Laboratory, Peking University)
- PKU-NetSys-LAB (NetSys Lab, Peking University)

Thanks to all the following [developers](https://github.com/jd-opensource/xllm/graphs/contributors) who have contributed to xLLM.

<a href="https://github.com/jd-opensource/xllm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/xllm" />
</a>

---

## 8. License
[Apache License](LICENSE)

#### xLLM is provided by JD.com 
#### Thanks for your Contributions!
