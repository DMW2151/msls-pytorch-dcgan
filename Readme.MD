# MSLS-DCGAN

## About

This repository contains code for training an implementation of Ian Goodfellow's original `DCGAN` model<sup>1</sup> on data sourced from Mapillary's Street Level Sequences dataset. Although this paper is quite old, I felt that it's worth taking a look at because it's a foundational paper in the field and can be implemented in just a few hundred lines of PyTorch. The code in this repo has been instrumented to run on either Gaudi accelerated or Nvidia GPU instances.

I would encourage you to refer to my full write-up [here](https://gan.dmw2151.com/trained-a-gan.html), which contains details on:

- Theory Behind DCGAN
- Discussion of MSLS Dataset & Data Preparation
- Discussion of AWS Architecture
- Modifications for Training on Gaudi (`DL1`) and Nvidia GPU (`P`)
- DCGAN Results
- Comparative Results w.r.t Cost and Hardware Performance (`DL1` v. `P`)

## Citations

**[1]** *"Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014.*
