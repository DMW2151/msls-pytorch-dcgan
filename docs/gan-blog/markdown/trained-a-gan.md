---
title: Generative Street-Level Imagery on DL1 Instances
author: Dustin Wilson
date: January 29, 2022
---

--------

<center>
    <figure class="image">
        <img src="./images/gan/001.gif" height="auto" width="188" style="padding: 20px; border-radius: 2px">
        <img src="./images/gan/002.gif" height="auto" width="188" style="padding: 20px; border-radius: 2px">
        <img src="./images/gan/003.gif" height="auto" width="188" style="padding: 20px; border-radius: 2px">
        <i><figcaption style="font-size: 12px;">Nowhere, USA - Experimental Output - Scenes created by interpolating between sequences of generated frames</figcaption></i>
    </figure>
</center>

For a few months now, I've wanted to create something like [ThisPersonDoesNotExist](https://thispersondoesnotexist.com/) for street scenes. Luckily, the [AWS Deep Learning Challenge](https://amazon-ec2-dl1.devpost.com) gave me an excuse to do so. At a high level, my project involved re-implementing elements of two foundational papers in generative computer vision and then training that model on over 1.1 million street-level images.

It's not a novel idea, but enough work has been done in this field that I was able to read up on the literature, implement generative models, and reason about architectural and performance tradeoffs. Critically, the challenge encouraged participants to use AWS' `DL1` instances to scale deep learning model training on HPUs. With that in mind, I instrumented my code to train on both GPU and Gaudi accelerators, and then performed a comparative analysis of performance across training environments.

- [Theory and Background](#Theory-and-Background)
- [Mapillary Street Level Imagery Data](#Mapillary-Street-Level-Imagery-Data)
- [DCGAN Results](#DCGAN-Results)
- [AWS System Architecture](#AWS-System-Architecture)
- [Evaluating a First Training run on GPU Instances](#Evaluating-a-First-Training-run-on-GPU-Instances)
- [Modifications for Training on Gaudi Accelerated Instances](#Modifications-for-Training-on-Gaudi-Accelerated-Instances)
- [Comparative Performance](#Comparative-Performance)
- [Appendix 1 - Comparable Instance Selection](#Appendix-1---Comparable-Instance-Selection)
- [Appendix 2 - PIL Benchmarks](#Appendix-2---PIL-Benchmarks)
- [Citations](#Citations)

--------

### Theory and Background

In this project I re-implement elements of Ian Goodfellow's [Generative Adversarial Networks (2014)](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)<sup>1</sup> and Alec Radford's [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks (2016)](https://arxiv.org/pdf/1511.06434.pdf)<sup>2</sup> papers in PyTorch. Both papers are concerned with the development of GANs, Generative Adversarial Networks.

Before discussing specific elements of the project, let's discuss the *way* GANs work. Put simply, GANs consist of two competing functions. A generator (`G`) tries to create believable data and a discriminator (`D`) tries to maximize the probability it correctly classifies real and generated data.

**Assume the following variables:**

- `X` &mdash; Input data, in our case, an image with size `(3 x 64 x 64)`
  
- `D(X)` or `D` &mdash; Discriminator network which outputs the probability that an input, `X`, is real.

- `G(Z)` or `G` &mdash; Generator network that deterministically creates data in the shape of `X`. In practice, an image with size `(3 x 64 x 64)`.
  
- `Z` &mdash; Random noise to seed the generator. In practice, a `(1 x 100)` vector drawn from a standard normal distribution.
  
- `D(G(Z))` &mdash; Given an output of the generator, the probability that the discriminator believes the image to be real. A high `D(G(Z))` suggests the generator has "tricked" the discriminator.

The critical steps in each training iteration involve measuring the values of the following terms. For the formula-inclined, the GAN is simply maximizing the following function:

<center>`min​`<sub>`G`</sub>`max​`<sub>`V`</sub>`(D,G) = E`<sub>`x∼pdata​(x)`</sub>​`[logD(x)] + E`<sub>`z∼pz​(z)​`</sub>`[log(1−D(G(z)))]`</center>

- `E`<sub>`x∼pdata​(x)`</sub>​`[logD(x)]` &mdash; The expected value of `D`'s predictions when given samples from the real batch. Remember, `D(x)` produces a probability, thus a perfect discriminator would return values near *0*.

- `E`<sub>`z∼pz​(z)​`</sub>`[log(1−D(G(z)))]` &mdash; The expected value of `D`'s prediction when given samples produced from `G(Z)`, Because all images in this batch are fake, a better discriminator would predict a lower `D(G(Z))`, also returning values near *0*.

In the DCGAN paper, the method by which this function is maximized is by putting batches of images through `D` and `G`, where both are convolutional neural networks with a specific layer structure.

<center>
    <figure>
        <img style="padding-top: 20px;" align="center" width="600" src="./images/translation/gan.png">
        <i><figcaption style="font-size: 12px;">DBGAN Generator Architecture -  As diagramed by Radford, et. al <sup>4<sup></figcaption></i>
    <figure>
</center>

At a low-level, it's difficult to describe all of the internal consequences of using `PyTorch` rather than the specific packages the authors used. At a high level, I made the following notable changes:

- Choose `AdamW`/`FusedAdamW` as an optimizer function over `SGD`. *Goodfellow, et al.* use a custom `SGD` [implementation](https://github.com/goodfeli/adversarial/blob/master/sgd.py) that is a patched version of pylearn2's `SGD`. Instead, I elected for a built-in PyTorch optimizer, `AdamW`. As an added benefit, Habana offers their own `FusedAdamW` implementation that should perform quite well on the Gaudi instances.

- In *Goodfellow, et al.*, the authors use the procedure described below to estimate the relative performance of multiple generative methods. Rather than using this procedure to evaluate other models, I implemented it for comparing intra-model progress across epochs, see [results](#DCGAN-Results) for a deeper discussion of model validation:
  
    > We estimate probability of the test set data under *Pg* by fitting a Gaussian Parzen window to the samples generated with G and reporting the log-likelihood under this distribution. The σ parameter of the Gaussians was obtained by cross validation on the validation set. This procedure was introduced in Breuleux et al. [7] and used for various generative models for which the exact likelihood is not tractable.

- I remove the final `Sigmoid` layer from `D`. Typically a binary classification problem like the one `D` solves would use [Binary Cross Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy) (`BCELoss`). The way that PyTorch optimizes for mixed-precision operations required I switch to `BCEWithLogitLoss`, a loss function that expects logits (`L∈(−∞,∞)`) rather than probabilities (`p∈[0,1]`). In effect, this change moves the `Sigmoid` from the network to part of the loss function.

--------

### Mapillary Street Level Imagery Data

<center>
    <figure>
    <img alt="training_samples_eu" style="padding-top: 20px;" align="center" width="600" src="./images/translation/train_samples_eu.png">
    <i><figcaption style="font-size: 12px;" >Training Samples From MSLS - Cropped and Transformed</figcaption></i>
    <figure>
</center>

Throughout this project, I used Mapillary's Street-Level Sequences data (MSLS). Mapillary provides a platform for crowd-sourced maps and street-level imagery, and publishes computer vision research using data collected from this platform. Mapillary has made this and other data publicly available for [download](https://www.mapillary.com/dataset/places) (**Note**: [GH Issue](https://github.com/mapillary/mapillary_sls/issues/23)). In total, MSLS contains 1.6 million images from 30 major cities on six-continents and covers different seasons, weather, daylight conditions, structural settings, etc.

The model presented here was trained on a sample of ~940,000 images. The remaining images were reserved for hyperparameter tuning, cross-validation, model evaluation, etc. The figure below shows an estimated count of images included in model training.

| Metro Area    | % of Sample | Approx. Count |
|:--------------|:-----------:|----------:|
| Amman         |       0.14% |     1,702 |
| Amsterdam     |       1.37% |    16,487 |
| Austin        |       1.90% |    22,847 |
| Bangkok       |       3.26% |    39,055 |
| Boston        |       1.27% |    15,204 |
| Budapest      |      17.67% |   212,015 |
| Goa           |       1.11% |    13,307 |
| Helsinki      |       1.75% |    20,978 |
| London        |       0.65% |     7,755 |
| Manila        |       0.53% |     6,416 |
| Melbourne     |      15.58% |   186,908 |
| Moscow        |      18.14% |   217,594 |
| Nairobi       |       0.06% |       725 |
| Ottawa        |      12.09% |   145,063 |
| Paris         |       1.62% |    19,416 |
| Phoenix       |      12.56% |   150,642 |
| Sao Paulo     |       4.65% |    55,793 |
| San Fransisco |       0.43% |     5,133 |
| Tokyo         |       3.49% |    41,845 |
| Toronto       |       1.27% |    15,176 |
| Trondheim     |       1.07% |    12,888 |
| Zurich        |       0.51% |     6,081 |
| **Total**         |             | **1,199,556** |
Table: Training Sample By Metro Area

Because the authors who developed MSLS for their [research](https://research.mapillary.com/publication/cvpr20c)<sup>3</sup> were specifically interested in place-recognition, the data is organized such that images of the same physical location appear multiple times under different conditions. The images from these sequences are very highly correlated and reduce the diversity of the training set far more than a single repeated image.

Originally, I was hoping to train a suite of metro-area models, but the effect of individual sequences was too pronounced and the model often reproduced images from the training set. I did some custom filtering to reduce the contribution of individual sequences, but found the most effective strategy was simply adding more metropolitan areas and converging on a single model.

The effect of multi-image sequences was further reduced by applying random transformations on each image. MSLS contains images up to `(3 x 640 x 480)`. Because the original DCGAN paper expects `(3 x 64 x 64)` images, I had leeway to apply cropping, down-scaling, and horizontal translations to all images before passing them through the model. Given the large image shown below, the model could receive any of the variations presented on the right.

<center>
    <figure>
    <img alt="nyc_sample_imgs" style="padding-top: 20px;" align="center" width="600" src="./images/translation/nyc_img_transformed_samples.png">
    <i><figcaption style="font-size: 12px;" >Sample Transformations - All images are shifted, center-cropped, and then scaled to `3 x 64 x 64`<sup>4</sup> </figcaption></i>
    <figure>
</center>

--------

### DCGAN Results

- Results
- Training Progress
- Loss
- Model Performance and Evaluation

--------

### AWS System Architecture

<center>
    <figure>
    <img alt="training_samples_eu" style="padding-top: 20px;" align="center" width="600" src="./images/infra/arch.png">
    <i><figcaption style="font-size: 12px;" >Simplified Model Training Architecture</figcaption></i>
    <figure>
</center>

All infrastructure for this project is hosted on AWS. If you'd like a user-guide for deploying the architecture yourself, I'd direct you to my infrastructure [repo](https://github.com/DMW2151/msls-infra). All training resources run in a single VPC with two subnets (1 public, 1 private) in the same availability zone. I deployed the following instances to the VPC's private subnet and accessed them via SSH through a jump-instance deployed to the public subnet.

- **training-prod** &mdash; An EC2 instance for running deep learning models, either `DL1` or a cost-comparable GPU instance (`P`-type). In either case, the instance is running a variant of the AWS Deep Learning AMI. Of course, you can construct your own conda environment, container, or AMI for your specific needs.
  
- **training-nb** &mdash; A small Sagemaker instance used for interactive model development, model evaluation, and generating plots.
  
- **metrics** &mdash; A small EC2 instance used to host metrics containers. This machine ran:
  - [Tensorboard](https://www.tensorflow.org/tensorboard) &mdash; A tool for visualizing *machine learning metrcs* during training.
  - [Grafana](https://grafana.com/) &mdash; An analytics and monitoring tool. I configured Grafana to visualize *machine-level* metrics from our training instances.

Each of these instances has access to an AWS Elastic Filesystem (EFS) for saving model data (e.g. checkpoints, plots, traces, etc.). Using EFS saved me hours of data transfer in development and allowed me to pass model checkpoints between machines (i.e. between *training-prod* and *training-nb*). Regrettably, an EFS file system can only drive up to 150 KiB/s per GiB of read throughput. With my filesystem using under 100GB, this left me with a paltry ~8MB/s data transfer.

To alleviate this issue, I downloaded the MSLS data to a `gp3` volume that I provisioned with high (8000) IOPS and throughput (1000 MiB/s). I can easily attach and detach it from separate training instances as as needed. Anecdotally, this choice led to a *2000%* speed up in time until first training iteration. Although EBS is more expensive, the decision paid for itself by saving hours of idle GPU/HPU time.

--------

### Evaluating a First Training run on GPU Instances

I started with a PyTorch model running on a GPU (`V100`) before instrumenting it to run on the HPU. I wanted to make sure that I could do a fair comparison of the two, and that meant ensuring I was optimizing (within reason) for either platform. To validate that the model was sufficiently tuned for the GPU, I referred to the metrics generated by running my model in profiler mode, instance metrics sent to Grafana, and those produced by `nvidia-smi` (see: [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface). With these metrics, over the course of a few hours I was able to go from quite poor to tolerable performance.

- **Batch Size** &mdash; This was low-hanging fruit. Independent of the other changes, the right choice of batch size sped up overall execution time by ~80%.

- **Minimize CUDA Copies** &mdash; Training statistics, outputs, labels, etc. were being haphazardly moved to and from the GPU! I can collect and display them at the end of the epoch.

- **Using AMP** &mdash; Automatic Mixed Precision (AMP) allows for model training to run with FP16 values where possible and F32 where needed. This allows for lower memory consumption and faster training time. It also opens the door for me to use Habana's mixed precision [modules](https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html#pytorch-mixed-precision-training-on-gaudi) when I move over to the `DL1` instance.

- **Distributed Data Processing** &mdash; In isolation, distributed data processing doesn't improve the model's training performance, but it does lend towards a more robust training environment. Although this is a problem that uses a moderate of small images, I still wanted to instrument my code to run across multiple GPUs (and nodes).

Looking at the first chart below, *PyTorch Profiler - GPU Execution Summary*, it would seem I was quite close to "perfect" GPU utilization. Unfortunately, the second graph reveals a fundamental problem in my profiling strategy at the time. The sections profiled didn't include the dataloader steps!

|                           |
|:-------------------------:|
| *Figure 1.1 - PyTorch Profiler - GPU Execution Summary* |
| ![OK](./images/training/big_batch_good.png) |

|                           |
|:-------------------------:|
| *Figure 1.2 - Grafana - GPU Utilization Rates* |
| ![Bad GPU](./images/training/gpu_poor.png) |

At this point things got quite difficult. I tried tweaking the number of dataloader workers and their pre-fetch factors, no luck. I tried generating an hd5 dataset from my images and writing my own dataloader, again, no luck. I even tried installing a [SIMD fork of PIL](https://github.com/uploadcare/pillow-simd) to increase image processing performance. Unfortunately, none of it made a meaningful difference on the `V100`. I strongly suspected it was the dataloader code that was the bottleneck and did a few sanity checks (see [Appendix 2](#Appendix-2---PIL-Benchmarks)) to make sense of things.

I did some research into [GPU profiling](https://pytorch.org/blog/pytorch-profiler-1.9-released/) and learned that GPU utilization is a coarse metric and I was probably already in a OK place from a performance perspective.

> Estimated Achieved Occupancy (Est. Achieved Occupancy) is a layer deeper than Est. SM Efficiency and GPU Utilization for diagnosing performance issues. ... As a rule of thumb, good throughput gains can be had by improving this metric to 15% and above. But at some point you will hit diminishing returns. If the value is already at 30% for example, further gains will be uncertain.

This low GPU utilization was still a bit unsettling, but my Est. Achieved Occupancy was good and the standard `pytorch.DataLoader` would stay in the code. Finally, I did a full "GPU" run on a multi-GPU instance (`p3.8xlarge`, 4 x `V100`) and I could  move along to training on the Gaudi-accelerated instances satisfied that I gave the GPU a fair shake.

|                           |
|:-------------------------:|
| *Figure 1.3 - GPU Training - PyTorch Profile - P3.8xLarge* |
| ![OK](./images/training/p3_8xlarge_profile.png) |

--------

### Modifications for Training on Gaudi Accelerated Instances

I started with a standard PyTorch model running on the GPU before instrumenting it with the code to run on the HPU. Migrating a model to run on HPUs require some changes, most of which are highlighted in the migration [guide](https://docs.habana.ai/en/latest/Migration_Guide/Migration_Guide.html#porting-simple-pyt-model). In general, a few changed imports allow the PyTorch Habana bridge to drive the execution of deep learning models on the Habana Gaudi device. Specifically, I made the following changes for the Gaudi accelerated instances:

- Swap out a standard `pytorch.DataLoader` for `habana_dataloader.HabanaDataLoader`. Under the right [circumstances](https://docs.habana.ai/en/v1.1.0/PyTorch_User_Guide/PyTorch_User_Guide.html#habana-data-loader), `HabanaDataLoader` can yield better performance that the native `DataLoader`. Even without acceleration, I can still use the `HabanaDataLoader` with a high `num_workers` parameter to quickly shuttle data onto the device.
  
- Use `Lazy Mode`. [Lazy Mode](https://docs.habana.ai/en/v1.1.0/PyTorch_User_Guide/PyTorch_User_Guide.html#lazy-mode) provides the SynapseAI graph compiler the opportunity to optimize the device execution for multiple ops.
  
- Use `FusedAdamW` over `AdamW`. `FusedAdamW` can batch the element-wise updates applied to all the model’s parameters into one or a few kernel launches rather than a single kernel for each parameter. This is a custom optimizer for Habana devices and should yield some performance improvements over `AdamW`.

--------

### Comparative Performance

- Hardware and Cost-To-Train

--------

### Appendix 1 - Comparable Instance Selection

Using [instances.vantage.sh](https://instances.vantage.sh/) and `aws describe-instances`, I aggregated data for all EC2 instances available in `us-east-1` with between 2 and 8 GPUs. These machines range from those with GPUs that are designed for graphics workloads (e.g. `G3` instances with Tesla `M60`s) to top-of-the line training instances (e.g. `P4` instances with `A100`s). I relied exclusively on Nvidia's most recent [resnext-101 benchmarks](https://developer.nvidia.com/deep-learning-performance-training-inference) as a proxy for my model's performance. On price, `p3.8xlarge` instances are the most similar to the `DL1` and offer 4 `V100`. Although `g4dn.12xlarge`(`T4`) and `p2.8xlarge` (`K80`) instances are priced well relative to their performance, I elected to only run a full test on the `p3.8xlarge`.

| API Name      | Memory (GiB) | VCPUs | GPUs | GPU Model             | GPU Mem (GiB) |   $/Hr |
|---------------|--------------|-------|------|-----------------------|---------------|--------|
| g3.8xlarge    |          244 |    32 |    2 | NVIDIA Tesla M60      |            16 |   2.28 |
| g3.16xlarge   |          488 |    64 |    4 | NVIDIA Tesla M60      |            32 |   4.56 |
| p2.8xlarge    |          488 |    32 |    8 | NVIDIA Tesla K80      |            96 |   7.20 |
| g4dn.12xlarge |          192 |    48 |    4 | NVIDIA T4 Tensor Core |            64 |   3.91 |
| g4dn.metal    |          384 |    96 |    8 | NVIDIA T4 Tensor Core |           128 |   7.82 |
| g5.12xlarge   |          192 |    48 |    4 | NVIDIA A10G           |            96 |   5.67 |
| g5.24xlarge   |          384 |    96 |    4 | NVIDIA A10G           |            96 |   8.14 |
| g5.48xlarge   |          768 |   192 |    8 | NVIDIA A10G           |           192 |  16.29 |
| p3.8xlarge    |          244 |    32 |    4 | NVIDIA Tesla V100     |            64 |  12.24 |
| p3.16xlarge   |          488 |    64 |    8 | NVIDIA Tesla V100     |           128 |  24.48 |
| p3dn.24xlarge |          768 |    96 |    8 | NVIDIA Tesla V100     |           256 |  31.21 |
| p4d.24xlarge  |         1152 |    96 |    8 | NVIDIA A100           |           320 |  32.77 |
Table: Table A.1.1 - Possible Comparable GPU Instances

### Appendix 2 - PIL Benchmarks

I narrowed down the source of the drops in GPU utilization to the dataloader being slow relative to the GPU. Every batch is doing thousands of `PIL.open()` calls ([source](https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py#L245-L249), if these calls are causing the slowdown, we should be able to see a huge amount of stress on the disk during the loader step.

- **Let's just use a worse GPU!** &mdash; I spun up a `p2.8xlarge` with 8 `K80`s to see if the weaker GPU would produce nicer utilization metrics. In theory, if the GPU is the bottleneck instead of the dataloader, I won't see these periodic dips. This is a bit of a vanity metric and I have no interest in doubling my training costs for vanity's sake, but the charts below confirm my hypothesis. This was an excellent discovery!

|                           |
|:-------------------------:|
| *Figure A2.1.1 - GPU Training - GPU Usage - P2.8xLarge* |
| ![OK](./images/training/vanity_gpu.png) |

- **Why not profile the disk?** &mdash;  Back on the `p3.2xlarge`, I figured I should profile the disk to see what was going on during the utilization drops. I thought a maxed-out `gp3` would have been adequate, but maybe I should have sprung for the `io1` or `io2`. In figure *A2.2.1 - XXXX*, you can see the results of `atop` and `nvidia-smi` during a training run. When the GPU is at low utilization. the disk where `MSLS` is mounted (`/dev/xvdh`) is **working!**.

|                           |
|:-------------------------:|
| *Figure A2.1 - GPU Training - Atop + Nvidia SMI Profile - P3.8xLarge* |
| ![OK](./images/training/disk_saturated.png) |

Thinking about it in retrospect, this all makes sense. We're opening images that are `(3 x 360 x 480)` and the GPU is doing some light calculations to resize and re-color them, but then running expensive convolutions on images that are just `(3 x 64 x 64)`.

--------

### Citations

**<sup>1</sup>** *"Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014.*

**<sup>2</sup>** *Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).*

**<sup>3</sup>** *F. Warburg, S. Hauberg, M. Lopez-Antequera, P. Gargallo, Y. Kuang, and J. Civera. Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020*

**<sup>4</sup>** *File:NYC 14th Street looking west 12 2005.jpg. (2020, September 13). Wikimedia Commons, the free media repository. Retrieved 23:09, January 25, 2022 from https://commons.wikimedia.org/w/index.php?title=File:NYC_14th_Street_looking_west_12_2005.jpg&oldid=457344851* 
