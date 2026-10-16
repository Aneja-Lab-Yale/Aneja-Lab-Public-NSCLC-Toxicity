# Third-party acknowledgments

The CT networks use Meta's DINOv3 ViT-S/16, through the `timm` implementation. Redistributed DINOv3 weights and derivatives are subject to the complete [DINOv3 license](DINOV3_LICENSE.md), which also accompanies the release assets. Acknowledge DINOv3 in research using these materials.

- [DINOv3 repository](https://github.com/facebookresearch/dinov3)
- [DINOv3 paper, Siméoni et al., 2025](https://arxiv.org/abs/2508.10104)
- [timm implementation and model card](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)

Python dependencies retain their own licenses. No original CT scans, clinical source files, patient identifiers, embeddings, or manuscript patient-level outputs are included in this public repository.

Automatic lung masks use the free `total` task of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), pinned to package 2.13.0, with [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 2.7.0. Cite the [TotalSegmentator paper](https://doi.org/10.1148/ryai.230024) and nnU-Net when using these tools. Their code and downloaded weights retain their upstream licenses; they are not redistributed in this Git repository.

The real CT example is downloaded directly from 3D Slicer. Its provenance, checksum, upstream terms, and fictional clinical inputs are documented in [PUBLIC_CT.md](PUBLIC_CT.md). The MIT synthetic-data license does not replace the public image source’s terms.
