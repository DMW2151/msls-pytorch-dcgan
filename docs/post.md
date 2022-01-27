## Baseline Model

Before discussing modifications, let's discuss the *"way"* this model works.

**Assume the following:**

- `Z` &mdash; Latent vector sampled from a standard normal distribution (`1 x 100`)
  
- `G(Z)` &mdash; Generator network which maps the latent vector, `Z`, to data-space. In practice, an image w. size `3 x 64 x 64`
  
- `X` &mdash; Vector (image) with size `3 x 64 x 64`

- `D(X)` &mdash; Discriminator network which outputs the probability that an input, `X`, is *real* (i.e. not an output of `G(Z)`)
  
- `D(G(Z))` &mdash; The probability that the output of the generator `G` is a real image.

Put simply, `G` tries to create believable images from the latent input vector and `D` tries to maximize the probability it correctly classifies real (from data) and fake images (produced from `G(Z)`). Throughout the code there are additional references to specific loss metrics, please see `gaudi_dcgan.py` for a description of those values.

The network's architecture is unchanged from the original paper and is diagramed below:

<center>
    <figure>
        <img style="padding: 20px;" align="center" width="600" src="./docs/images/translation/gan.png">
        <figcaption>DBGAN Architecture - As diagramed by <i>Radford, et. al <sup>4<sup></i></figcaption>
    <figure>
</center>

At a low-level, it's difficult to describe all of the consequences of using `PyTorch` (at least without a deep understanding of PyTorch internals). At a high level, I made the following notable changes:

- Choose `AdamW`/`FusedAdamW` as an optimizer function over `SGD`. *Goodfellow, et al.* use a custom `SGD` [implementation](https://github.com/goodfeli/adversarial/blob/master/sgd.py) in their paper that is a patched version of `pylearn2`'s `SGD` function. Instead, I elected for a (slightly) more modern optimizer. `AdamW` is a solid, general-purpose optimizer. As an added benefit, Habana offers their own `FusedAdamW` implementation that should perform quite well on the Gaudi units.
  
- The original paper uses `CIFAR-10`, `MNIST`, and `TFD` to evaluate performance against several other generative methods. My project is not interested in demonstrating the validity of the architecture and does not report any comparative metrics. However, I use the methods described below to compare the same model across epochs:

    > Estimate probability of the test set data by fitting a Gaussian Parzen window to the samples generated with G and reporting the log-likelihood under this distribution.

As I intended to test this model on both Nvidia GPUs and Gaudi acceleators, I should also note the following changes between the model's execution on the two machines:

- When running on Gaudi processors, swap out a standard `pytorch.DataLoader` for `habana_dataloader.HabanaDataLoader`. Under the right [circumstances](https://docs.habana.ai/en/v1.1.0/PyTorch_User_Guide/PyTorch_User_Guide.html#habana-data-loader), `HabanaDataLoader` can yield better performance that the native `DataLoader`.
  
- When running on Gaudi processors, use `Lazy Mode`. [Lazy Mode](https://docs.habana.ai/en/v1.1.0/PyTorch_User_Guide/PyTorch_User_Guide.html#lazy-mode) provides the SynapseAI graph compiler the opportunity to optimize the device execution for multiple ops.
  
- When running on Gaudi processors, use `FusedAdamW` over `AdamW`. `FusedAdamW` can batch the element-wise updates applied to all the model’s parameters into one or a few kernel launches rather than a single kernel for each parameter.

## Training Infrastructure

I've provided a full description of the AWS system architecure for training [here](https://github.com/DMW2151/msls-dcgan-infra).
Broadly, I train the model once on a `DL1.24xlarge` instance and once on a `ml.p3.8xlarge` instance.

The `ml.p3.8xlarge` instance is not a meant to be a perfect comparison for the `DL1`. It's simply a similarly-priced GPU instance that a team may consider for training deep-learning models. I perform an analysis of comparable instances in my [main post](https://dmw2151.com/trained-a-gan).

## Mapillary Street Level Sequences Data

Models trained on both `DL1` and `p3` instances use a subset of the Mapillary Street-Level Sequences dataset (MSLS). Mapillary, a subsidiary of Facebook, primarily provides a platform for crowd-sourced maps and street-level imagery. This dataset is available for download [here](https://www.mapillary.com/dataset/places)<sup>2</sup>.

In total, MSLS contains 1.6 million images from 30 major cities on six-continents. Image sizes range from `(3 x 256 x 256)` to `(3 x 640 x 480)`. Because the original `DCGAN` paper expects `(3 x 64 x 64)` images, the larger MSLS sizes gave me leeway to apply cropping, down-scaling, and multiple horizontal translations to almost all images in the sample.

I considered using `(3 x 128 x 128)` images for this project and adding an additional `2DConv`/`ReLu` layer to both the Discriminator and Generator networks to handle for the newly sized images. However, this seemed like a significant deviation from my stated goal, and I elected to train on `(3 x 64 x 64)` as in the original paper. In the last 8 years *many* methods have been developed to handle `(3 x 128 x 128)` (and *much* larger [images](https://github.com/lucidrains/lightweight-gan)).

The model presented here was trained on a sample of ~700,000 images from the MSLS data. An additional ~300,000 images were held-out to be used in model evaluation.

```bash
# Training Sample
| City      | Train-Img |
|-----------|-----------|
| Austin    |    28,462 |
| Bangkok   |    40,125 |
| Budapest  |    45,800 |
| Helsinki  |    15,228 |
| London    |     5,983 |
| Manila    |     5,378 |
| Melbourne |   189,945 |
| Moscow    |   171,878 |
| Paris     |     9,503 |
| Phoenix   |   106,221 |
| Sao Paulo |    35,096 |
| SF        |     4,525 |
| Trondheim |     4,136 |
| Zurich    |     2,991 |
| Total     |   665,271 |
```

```bash
# Holdout Sample
| City      | HO-Imgs   |
|-----------|-----------|
| Amman     |     1,811 |
| Amsterdam |     7,908 |
| Boston    |    14,037 |
| Goa       |     5,735 |
| Kampala   |     2,069 |
| Nairobi   |       887 |
| Ottawa    |   123,296 |
| Phoenix   |    50,256 |
| Saopaulo  |    19,002 |
| Tokyo     |    34,836 |
| Toronto   |    12,802 |
| Trondheim |     5,028 |
| Total     |   277,667 |
```

### Image Translations

As noted in the previous section, MSLS images are significantly larger than those DCGAN accepts as an input. The following code is an annotated excerpt from `run_gaudi_dcgan.py` and applies a transformation from a single raw image to an image used in the model.

```python
# Create pytorch.Data.ImageFolder from `DATAROOT`; Focus on transformations
data = dset.ImageFolder(
    root=DATAROOT,
    transform=transforms.Compose([
        # (1) Middle 1/2 of the Image is Most "Interesting", especially towards the edges, things like
        # sidewalks, trees, etc. => Apply a random horizontal shift of (-30%, 30%) * orig_W
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.0)),

        # (2) Use the Middle 256 x 256) of the resulting, shifted image
        transforms.CenterCrop(IMG_SIZE * 4),

        # (3) Downsize to 64 x 64 -> NOTE: The downsampling/interpolation of PIL images applies
        # antialiasing by default, this (seems to be) a good choice, proceed...
        transforms.Resize(IMG_SIZE),

        # (4) Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.ToTensor(),

        # (5) Normalize colors -> Normalizes a tensor image with mean and standard deviation.
        # In this case, we apply mean and STD of 0.5, 0.5 for each channel (RGB).
        transforms.Normalize( (0.5, 0.5, 0.5), (0,5, 0.5, 0.5) )
    ]),
)
```

In practice, this could mean an image<sup>3</sup> like the one shown below could be transformed to look like any of those on the right (scale preserved). The second image shown below is a sample of 16 training images derived from the from the MSLS dataset.

<center>
    <figure>
    <img alt="nyc_sample_imgs" style="padding-top: 20px;" align="center" width="600" height="400" src="./docs/images/translation/nyc_img_transformed_samples.png">
    <figcaption>Sample Transformations (NYC)</figcaption>
    <figure>
</center>

<center>
    <figure>
    <img alt="training_samples_eu" style="padding-top: 20px;" align="center" width="600" src="./docs/images/translation/train_samples_eu.png">
    <figcaption>Training Samples From MSLS (Global) </figcaption>
    <figure>
</center>


## Supplemental Links & Citations

### Links

- [DCGAN Code](https://github.com/goodfeli/adversarial)
- [MSLS Publication](https://research.mapillary.com/publication/cvpr20c)
- [MSLS Release Notes](https://blog.mapillary.com/update/2020/04/27/Mapillary-Street-Level-Sequences.html)


### Citations

**[1]** *"Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014.*

**[2]** *F. Warburg, S. Hauberg, M. Lopez-Antequera, P. Gargallo, Y. Kuang, and J. Civera. Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020*

**[3]** *File:NYC 14th Street looking west 12 2005.jpg. (2020, September 13). Wikimedia Commons, the free media repository. Retrieved 23:09, January 25, 2022 from https://commons.wikimedia.org/w/index.php?title=File:NYC_14th_Street_looking_west_12_2005.jpg&oldid=457344851* 

**[4]** *Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).*
