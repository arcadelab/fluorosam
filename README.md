
<div align="center">

# FluoroSAM

![Overview](/assets/overview.png)

</div>

Official repository for our paper, [*FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation*](https://arxiv.org/abs/2403.08059). Check back soon for public release of code, data, and model weights.

## Overview

Automated X-ray image segmentation would accelerate research and development in diagnostic and interventional precision medicine. Prior efforts have contributed task-specific models capable of solving specific image analysis problems, but the utility of these models is restricted to their particular task domain, and expanding to broader use requires additional data, labels, and retraining efforts. Recently, foundation models (FMs) – machine learning models trained on large amounts of highly variable data thus enabling broad applicability – have emerged as promising tools for automated image analysis. Existing FMs for medical image analysis focus on scenarios and modalities where objects are clearly defined by visually apparent boundaries, such as surgical tool segmentation in endoscopy. X-ray imaging, by contrast, does not generally offer such clearly delineated boundaries or structure priors. During X-ray image formation, complex 3D structures are projected in transmission onto the imaging plane, resulting in overlapping features of varying opacity and shape. To pave the way toward an FM for comprehensive and automated analysis of arbitrary medical X-ray images, we develop FluoroSAM, a language-aligned variant of the Segment-Anything Model, trained from scratch on 1.6M synthetic X-ray images from a wide variety of human anatomies, X-ray projection geometries, energy spectra, and viewing angles. FluoroSAM is trained on data including masks for 128 organ types and 464 non-anatomical objects, such as tools and implants. In real X-ray images of cadaveric specimens, FluoroSAM is able to segment bony anatomical structures based on text-only prompting with 0.51 and 0.79 DICE with point-based refinement, outperforming competing SAM variants for all structures. FluoroSAM is also capable of zero-shot generalization to segmenting classes beyond the training set thanks to its language alignment, which we demonstrate for full lung segmentation on real chest X-rays.

## Citation

If you find this work interesting, please consider citing [the paper](https://arxiv.org/abs/2403.08059):

```bibtex
@article{killeen2024fluorosam:,
	author = {Killeen, Benjamin D. and Wang, Liam J. and Zhang, Han and Armand, Mehran and Taylor, Russell H. and Osgood, Greg and Unberath, Mathias},
	title = {{FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation}},
	journal = {arXiv},
	year = {2024},
	month = mar,
	eprint = {2403.08059},
	url = {https://arxiv.org/abs/2403.08059v1}
}
```
