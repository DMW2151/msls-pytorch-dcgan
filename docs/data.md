# Data

## Mapillary Street Level Sequences Data

![sample](./images/sample.jpg)

Models trained on both `DL1` and `P3` instances use a subset of the Mapillary Street-Level Sequences dataset (MSLS). Mapillary, a subsidiary of Facebook, primarily provides a platform for crowd-sourced maps and street-level imagery. This dataset is available for download [here](https://www.mapillary.com/dataset/places).

In total, MSLS contains ~1.6 million images from 30 major cities on six-continents. In general, image sizes range from 256x256 to 640x480. Because the original DCGAN paper expects 64x64 images, these images gave me leeway to apply cropping, down-scaling, and multiple translations to almost all images in the sample.

The model presented here was trained on a sample of the MSLS using images from only the United States. The approximate distribution by metro area is below:

    | Metro-Statistical Area                       | Images (count) |
    |----------------------------------------------|----------------|
    | Miami-Fort Lauderdale-Pompano Beach, FL  MSA | 5,900          |
    | Austin–Round Rock-Georgetown MSA             | 28,460         |
    | Phoenix MSA                                  | 106,220        |
    | San Francisco-Oakland-Fremont, CA MSA        | 4,525          |

### Supplemental Links & Citation

- [Publication](https://research.mapillary.com/publication/cvpr20c)
- [Release Notes](https://blog.mapillary.com/update/2020/04/27/Mapillary-Street-Level-Sequences.html)

**[1]** *F. Warburg, S. Hauberg, M. Lopez-Antequera, P. Gargallo, Y. Kuang, and J. Civera. Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020*