# Real public CT example

The primary quickstart uses **3D Slicer CTChest**, the chest CT distributed by the official Slicer Sample Data module. The downloader retrieves the original file directly from its publisher, verifies the published SHA256, and records provenance in `data/public_ct/source.json`. No copy of our private cohort is used or distributed.

- [Sample Data description](https://www.slicer.org/wiki/SampleData)
- [Pinned source registration](https://github.com/Slicer/Slicer/blob/9ae94df99124986e65094520d5631baef32314e3/Modules/Scripted/SampleData/SampleData.py)
- [Direct public CT download](https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e)
- [Upstream Slicer license](https://github.com/Slicer/Slicer/blob/main/License.txt)

File: `CT-chest.nrrd`, 42,189,103 bytes. SHA256:

```text
4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e
```

Run:

```bash
python -m nsclc_toxicity download-demo-ct
python -m nsclc_toxicity infer-ct --manifest data/public_ct/clinical.csv --output results/public_ct
```

The public scan has 512 × 512 × 139 voxels and 0.761719 × 0.761719 × 2.5 mm spacing. Conversion preserves HU and spatial geometry; subsequent axis alignment matches LAS model inputs. The five lobe masks, seven QCT values, and twelve network hazard inputs are **computed from these voxels**, not copied from the synthetic dataset. [`public_ct_validation.json`](public_ct_validation.json) records an actual successful run.

**Clinical values are fictional.** The scan's SBRT history, tumor location/size/stage/histology, treatment timing/dose, and toxicity outcomes are not provided by this source. The example `clinical.csv` uses explicitly invented values. It contains no imaging features; the program computes those. It cannot establish the public subject's risk, toxicity prediction accuracy, or reproduce study AUCs.

In the notebook, inspect measured features, actual network outputs, and a CT/mask overlay. Clinical use requires independent validation and review of image coverage, orientation, and masks. `download-demo-ct` preserves an existing clinical CSV so it will not replace values you edited.
