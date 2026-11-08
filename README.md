# Foundation-model adaptation vs. task-specific training for medical image segmentation

Code for the paper *Foundation Model Adaptation versus Task-Specific Training for Medical Image Segmentation: Cross-Dataset Generalisation and Adaptation Capacity*

The study compares a DINOv2-Small encoder with a linear decoder, adapted under frozen, BitFit, LoRA (ranks 4–32) and full fine-tuning, against a plain and a residual U-Net trained from scratch under the nnU-Net recipe, in three domains (colonoscopy polyps, breast ultrasound, dermoscopy) with eight out-of-distribution test sets.

## Repository layout

| Folder | Contents |
|---|---|
| `notebooks/` | Training and evaluation notebooks, one per domain (`polyp.ipynb`, `breast.ipynb`, `skin.ipynb`). They download and prepare the data, train every configuration reported in the paper and save checkpoints. Outputs are stripped. |
| `evaluation/` | `extended_eval.py` recomputes DSC and the six further metrics (IoU, sensitivity, precision, HD95, ASSD, NSD) per image from saved checkpoints and checks the recomputed DSC against the reported values. `check_cvct.py` is the content-level overlap check between CVC-T and CVC-ColonDB. `busi_dup_check.py` is the perceptual-hash duplicate check on BUSI. |
| `analysis/` | Scripts that turn the per-image metrics into the tables and figures of the paper. |
| `results/` | Per-image metrics for every evaluated checkpoint (`metrics_<domain>.csv`), as produced by `evaluation/extended_eval.py`. |

## Requirements

Training and evaluation were run on Google Colab with a GPU; analysis runs on a CPU.

```bash
pip install -r requirements.txt
pip uninstall -y torchao   # conflicts with peft on current Colab images
```

`transformers` must be a 4.x release (4.57.6 was used): version 5 renames the DINOv2 attention modules, and the LoRA checkpoints will not load.

## Data

All datasets are public and are obtained from their original sources, cited in the paper; they are not redistributed here.

- **Polyp:** the standard PraNet split (Kvasir-SEG and CVC-ClinicDB for training; CVC-ColonDB, ETIS-LaribPolypDB and CVC-T held out), available from Zenodo record [5579392](https://zenodo.org/records/5579392) (`PolypDataset.zip`). The polyp notebook downloads it.
- **Breast:** BUSI for training; BrEaST and BUS-UCLM held out.
- **Skin:** ISIC 2018 for training; filtered subsets of ISIC 2016 and ISIC 2017, and PH2, held out.

The notebooks contain the preparation steps, including the removal of normal (empty-mask) BUSI images and the perceptual-hash deduplication of BUSI.

## Reproducing the paper

1. **Train.** Run the domain notebook on a GPU runtime. Checkpoints are written to `MyDrive/thesis_ckpts/` on Google Drive.
2. **Evaluate.** Upload `evaluation/extended_eval.py` next to the checkpoints, run the data-preparation cells of the notebook (no training cell), then:
   ```python
   import sys; sys.path.insert(0, '/content/drive/MyDrive/thesis_ckpts')
   import extended_eval as ee
   df = ee.run_domain('polyp', clean_sets, CKPT)   # or 'breast', 'skin'
   ```
   This writes `metrics_<domain>.csv`, a validation report, and `qualitative_<domain>.npz`.
3. **Tables and figures.** From the repository root:
   ```bash
   python analysis/tables.py        # main results, empty predictions, lesion size, resolution, capacity, strategies
   python analysis/make_capfig.py   # capacity figure
   python analysis/make_qual.py     # qualitative figure (needs qual/<domain>.npz from step 2)
   ```
   Outputs go to `outputs/tables/` and `outputs/figs/`. `tables.py` also produces the per-image distribution and regime figures.

Tables that report training-time results (learning-rate and configuration searches, seed repeats, budget and rank controls) come from the training logs in the notebooks.

## Checkpoints

Checkpoints are not included because of their size; they are available from the authors on request.

## Citation

If you use this code, please cite the paper (citation to be added on publication).

## License

MIT (see `LICENSE`).
