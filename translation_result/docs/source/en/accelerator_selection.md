<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 가속기 선택 [[accelerator-selection]]

분산 학습 중에는 사용할 가속기(CUDA, XPU, MPS, HPU 등)의 수와 순서를 지정할 수 있습니다. 이는 서로 다른 컴퓨팅 성능을 가진 가속기가 있을 때 더 빠른 가속기를 먼저 사용하고 싶은 경우에 유용할 수 있습니다. 또는 사용 가능한 가속기의 일부만 사용할 수도 있습니다. 선택 과정은 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)과 [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) 모두에서 작동합니다. Accelerate나 [DeepSpeed integration](./main_classes/deepspeed)는 필요하지 않습니다.

이 가이드는 사용할 가속기의 수와 사용 순서를 선택하는 방법을 보여줍니다.

## 가속기 수 [[number-of-accelerators]]

예를 들어, 4개의 가속기가 있고 처음 2개만 사용하고 싶다면 아래 명령을 실행하세요.

<hfoptions id="select-accelerator">
<hfoption id="torchrun">

`--nproc_per_node`를 사용하여 사용할 가속기 수를 선택합니다.

```bash
torchrun --nproc_per_node=2  trainer-program.py ...
```

</hfoption>
<hfoption id="Accelerate">

`--num_processes`를 사용하여 사용할 가속기 수를 선택합니다.

```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

</hfoption>
<hfoption id="DeepSpeed">

`--num_gpus`를 사용하여 사용할 GPU 수를 선택합니다.

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

</hfoption>
</hfoptions>

## 가속기 순서 [[order-of-accelerators]]
사용할 특정 가속기와 그 순서를 선택하려면 하드웨어에 적합한 환경 변수를 사용하세요. 이는 종종 각 실행에 대해 명령줄에서 설정되지만, `~/.bashrc`나 다른 시작 구성 파일에 추가할 수도 있습니다.

예를 들어, 4개의 가속기(0, 1, 2, 3)가 있고 가속기 0과 2만 실행하고 싶다면:

<hfoptions id="accelerator-type">
<hfoption id="CUDA">

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

GPU 0과 2만 PyTorch에서 "보이며" 각각 `cuda:0`과 `cuda:1`로 매핑됩니다.  
순서를 바꾸려면 (GPU 2를 `cuda:0`으로, GPU 0을 `cuda:1`로 사용):


```bash
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

GPU 없이 실행하려면:

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

`CUDA_DEVICE_ORDER`를 사용하여 CUDA 장치의 순서를 제어할 수도 있습니다:

- PCIe 버스 ID 순서 (`nvidia-smi`와 일치):

    ```bash
$hf_i18n_placeholder21export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ```

- 컴퓨팅 성능 순서 (가장 빠른 것부터):

    ```bash
    export CUDA_DEVICE_ORDER=FASTEST_FIRST
    ```

</hfoption>
<hfoption id="Intel XPU">

```bash
ZE_AFFINITY_MASK=0,2 torchrun trainer-program.py ...
```

XPU 0과 2만 PyTorch에서 "보이며" 각각 `xpu:0`과 `xpu:1`로 매핑됩니다.  
순서를 바꾸려면 (XPU 2를 `xpu:0`으로, XPU 0을 `xpu:1`로 사용):

```bash
ZE_AFFINITY_MASK=2,0 torchrun trainer-program.py ...
```


다음을 사용하여 Intel XPU의 순서를 제어할 수도 있습니다:

```bash
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

Intel XPU에서의 장치 열거 및 정렬에 대한 자세한 정보는 [Level Zero](https://github.com/oneapi-src/level-zero/blob/master/README.md?plain=1#L87) 문서를 참조하세요.

</hfoption>
</hfoptions>



> [!WARNING]
> 환경 변수는 명령줄에 추가하는 대신 내보낼 수 있습니다. 환경 변수가 어떻게 설정되었는지 잊어버리고 잘못된 가속기를 사용하게 될 수 있어 혼란을 야기할 수 있으므로 권장하지 않습니다. 대신, 같은 명령줄에서 특정 훈련 실행을 위해 환경 변수를 설정하는 것이 일반적인 관례입니다.
```