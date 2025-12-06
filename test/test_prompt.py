output = """
What do these sentences about Hugging Face Transformers (a machine learning library) mean in Korean? Please do not translate the word after a 🤗 emoji as it is a product name. Output only the translated result without any explanations or introductions.
```md
# Accelerator selection

During distributed training, you can specify the number and order of accelerators (CUDA, XPU, MPS, HPU, etc.) to use. This can be useful when you have accelerators with different computing power and you want to use the faster accelerator first. Or you could only use a subset of the available accelerators. The selection process works for both [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html). You don't need Accelerate or [DeepSpeed integration](./main_classes/deepspeed).

This guide will show you how to select the number of accelerators to use and the order to use them in.

## Number of accelerators

For example, if there are 4 accelerators and you only want to use the first 2, run the command below.

<hfoptions id="select-accelerator">
<hfoption id="torchrun">

Use the `--nproc_per_node` to select how many accelerators to use.

</hfoption>
<hfoption id="Accelerate">

Use `--num_processes` to select how many accelerators to use.

</hfoption>
<hfoption id="DeepSpeed">

Use `--num_gpus` to select how many GPUs to use.

</hfoption>
</hfoptions>

## Order of accelerators
To select specific accelerators to use and their order, use the environment variable appropriate for your hardware. This is often set on the command line for each run, but can also be added to your `~/.bashrc` or other startup config file.

For example, if there are 4 accelerators (0, 1, 2, 3) and you only want to run accelerators 0 and 2:

<hfoptions id="accelerator-type">
<hfoption id="CUDA">

Only GPUs 0 and 2 are "visible" to PyTorch and are mapped to `cuda:0` and `cuda:1` respectively.  
To reverse the order (use GPU 2 as `cuda:0` and GPU 0 as `cuda:1`):

To run without any GPUs:

You can also control the order of CUDA devices using `CUDA_DEVICE_ORDER`:

- Order by PCIe bus ID (matches `nvidia-smi`):

    

- Order by compute capability (fastest first):

    

</hfoption>
<hfoption id="Intel XPU">

Only XPUs 0 and 2 are "visible" to PyTorch and are mapped to `xpu:0` and `xpu:1` respectively.  
To reverse the order (use XPU 2 as `xpu:0` and XPU 0 as `xpu:1`):

You can also control the order of Intel XPUs with:

For more information about device enumeration and sorting on Intel XPU, please refer to the [Level Zero](https://github.com/oneapi-src/level-zero/blob/master/README.md?plain=1#L87) documentation.

</hfoption>
</hfoptions>

> [!WARNING]
> Environment variables can be exported instead of being added to the command line. This is not recommended because it can be confusing if you forget how the environment variable was set up and you end up using the wrong accelerators. Instead, it is common practice to set the environment variable for a specific training run on the same command line.
```
"""
