# Few-Shot Knowledge Distillation with Coreset Selection

Code for the experiments on **few-shot knowledge distillation** combined with **coreset selection** strategies on multiple image-classification benchmarks.

We compare several selection methods (Random baseline, Herding, K-Center Greedy, Few-Medoids, PCA-Guided Matching) used to pick a tiny per-class subset of the training set, on top of which a student network is distilled from a stronger teacher.

- **Datasets:** CIFAR-10, CIFAR-100, Flowers-102, Food-101
- **Teachers:** ResNet-34 / ViT-B/16
- **Students:** ResNet-18, ResNet-50, ViT-Small
- **Selection methods:** `Random`, `Herding_feature_vectors`, `KCenter_feature_vectors`, `FewMedoids_feature_vectors`, `PCA_Guided_Matching`

---

## 1. Repository layout

```
.
├── data/                  # raw datasets (auto-downloaded by torchvision)
├── split/                 # train / val index splits per dataset & seed
│   └── split_dataset.ipynb
├── model/                 # trained teachers and CE-only student baselines
│   └── fine_tunning.ipynb
├── selection/             # coreset selection per class
│   ├── feature_vectors.ipynb
│   ├── pca_feature_vectori_eigen_image.ipynb
│   └── <DATASET>/methods_<TEACHER>/<METHOD>/seed_<S>/split_<P>/k_<K>/
└── kd/                    # few-shot knowledge distillation
    ├── few_shot_kd.ipynb
    └── teacher_<TEACHER>/student_<STUDENT>/<DATASET>/seed_<S>/split_<P>/k_<K>/<METHOD>/
```

---

## 2. Pipeline (recommended order)

The four notebooks below must be run in order. Every notebook reads its configuration from constants defined at the top — change `DATASET_NAME`, `TEACHER_NAME`, `STUDENT_NAME`, `SEED`, `SPLIT`, `METHOD`, etc. there.

### Step 1 — Train / validation split
**Notebook:** [`split/split_dataset.ipynb`](split/split_dataset.ipynb)

Generates the train/val indices for the chosen dataset. Output:

```
split/<DATASET>/seed_<SEED>/split_<SPLIT>/
├── train_indices.npy
└── val_indices.npy
```

Example: `split/CIFAR10/seed_42/split_0.2/train_indices.npy`.

### Step 2 — Train the teacher (and the CE-only student baselines)
**Notebook:** [`model/fine_tunning.ipynb`](model/fine_tunning.ipynb)

Train each teacher once per dataset, and also train the **student CE-only baseline** (the chosen student — `ResNet18`, `ResNet50` or `ViT_Small` — trained on the full training split with cross-entropy only). The CE-only baseline accuracy is later used as the reference for the KD gain.

Outputs are saved in a folder whose name encodes the hyper-parameters and validation accuracy:

```
model/<MODEL_NAME>/<DATASET>/seed_<SEED>/split_<SPLIT>/
   val_acc=<…>%_val_loss=<…>_lr=<…>_wd=<…>_ep=<…>_bs=<…>_opt=<…>_sch=<…>/
       ├── model.pth
       ├── metrics.json
       ├── plot_accuracy.pdf
       ├── plot_loss.pdf
       └── …
```

Examples:
- Teacher: `model/ResNet34/CIFAR10/seed_42/split_0.2/val_acc=96.70%_val_loss=0.1164_lr=0.01_wd=0.0005_ep=100_bs=128_opt=SGD_sch=CosineAnnealingLR/model.pth`
- Student CE-only (ResNet-18): `model/ResNet18/CIFAR10/seed_42/split_0.2/val_acc=94.25%_val_loss=0.1993_lr=0.05_wd=0.0005_ep=100_bs=32_opt=SGD_sch=CosineAnnealingLR/model.pth`
- Student CE-only (ViT-Small): `model/ViT_Small/CIFAR100/seed_42/split_0.2/val_acc=…%_…/model.pth`

> The KD notebook reads its hyper-parameters automatically from the CE-only student's `metrics.json`, so make sure that folder exists for the student you intend to distil into.

### Step 3 — Coreset selection
**Notebooks:**
- [`selection/feature_vectors.ipynb`](selection/feature_vectors.ipynb) — for `Random`, `Herding_feature_vectors`, `KCenter_feature_vectors`, `FewMedoids_feature_vectors`
- [`selection/pca_feature_vectori_eigen_image.ipynb`](selection/pca_feature_vectori_eigen_image.ipynb) — for `PCA_Guided_Matching`

The **trained teacher is reused as a feature extractor** (logits / penultimate features). For each class, the chosen method selects `K` samples per class for every `K` in `K_LIST`.

**Important — values of K:**
- For **Flowers-102** use `K_LIST = [1, 2, 4, 8]` (the training split has very few images per class).
- For **CIFAR-10 / CIFAR-100 / Food-101** use `K_LIST = [1, 2, 4, 8, 16, 32, 64, 128]`.

**Important — Random must be run first when using K-Center Greedy:**
`KCenter_feature_vectors` is a *greedy* method seeded with one initial sample per class. We initialise it from the **Random** selection at `K=1`, so:

> Run `Random` selection **before** `KCenter_feature_vectors`. Otherwise at `K=1` K-Center has no seed and would degenerate to picking the same example as Random anyway.

Outputs are saved per class:

```
selection/<DATASET>/methods_<TEACHER>/<METHOD>/seed_<SEED>/split_<SPLIT>/k_<K>/
   ├── class_0_<name>_indices.npy
   ├── class_1_<name>_indices.npy
   └── …
```

Random is the only method stored under a flat seed root (no `methods_<TEACHER>` folder), since it doesn't depend on the teacher:

```
selection/<DATASET>/Random/seed_<S>/split_<SPLIT>/k_<K>/class_<id>_<name>_indices.npy
```

Examples:
- `selection/CIFAR10/methods_ResNet34/Herding_feature_vectors/seed_42/split_0.2/k_8/class_3_cat_indices.npy`
- `selection/CIFAR10/Random/seed_2/split_0.2/k_16/class_0_airplane_indices.npy`

### Step 4 — Few-shot knowledge distillation
**Notebook:** [`kd/few_shot_kd.ipynb`](kd/few_shot_kd.ipynb)

Distils the chosen student (`ResNet18`, `ResNet50` or `ViT_Small`) from the teacher using only the selected coreset, for every K in `K_LIST`. For methods with multiple selections (`Random`, `KCenter_feature_vectors`) it runs once per `seed in SEEDS = [0,1,2,3,4]` and aggregates mean/std. For deterministic methods (`Herding`, `FewMedoids`, `PCA_Guided_Matching`) the same coreset is used and the student is re-trained with 5 different training seeds.

Outputs:

```
kd/teacher_<TEACHER>/student_<STUDENT>/<DATASET>/seed_<SEED>/split_<SPLIT>/k_<K>/<METHOD>/
   val_acc=<…>%_val_loss=<…>_soft_weight=<…>_cr_weight=<…>_T=<…>_lr=<…>_wd=<…>_ep=<…>_bs=<…>_opt=<…>_sch=<…>/
       ├── model_<random|multi_seed|deterministic>_seed_<best_seed>.pth
       ├── metrics.json                       # KD config (hyper-params, transforms, weights, …)
       ├── random_metrics.json   |   multi_seed_metrics.json   |   deterministic_metrics.json
       ├── plot_accuracy.pdf
       ├── plot_loss.pdf
       ├── cm_test.pdf                        # confusion matrix (sparse for >10 classes)
       ├── per_class_test_top20.pdf           # only when NUM_CLASSES > 10
       ├── per_class_test_bottom20.pdf        # only when NUM_CLASSES > 10
       └── tsne_test.pdf
```

Examples:
- `kd/teacher_ResNet34/student_ResNet18/CIFAR10/seed_42/split_0.2/k_8/Herding_feature_vectors/val_acc=…/model_deterministic_seed_3.pth`
- `kd/teacher_ResNet34/student_ViT_Small/CIFAR100/seed_42/split_0.2/k_16/Random/val_acc=…/random_metrics.json`
- `kd/teacher_ViT_B_16/student_ResNet50/Flowers102/seed_42/split_0.2/k_4/FewMedoids_feature_vectors/val_acc=…/metrics.json`

The `*_metrics.json` aggregate file contains the per-run accuracies plus mean/std over the 5 runs.

---

## 3. Notes

- **K-Center seed dependency.** Always run `Random` selection before `KCenter_feature_vectors`. Without it the `K=1` initialisation has no seed sample and the greedy expansion cannot start (it would otherwise reduce to a Random pick).
- **K_LIST per dataset.** Use `[1, 2, 4, 8]` for **Flowers-102** (very small per-class training pool) and `[1, 2, 4, 8, 16, 32, 64, 128]` for the other datasets.
- **Teacher is also the feature extractor.** The same trained teacher checkpoint used for KD is the one whose features drive the selection — keep teacher and selection folders in sync (`methods_<TEACHER_NAME>`).
- **Reproducibility.** All notebooks call `set_seed(SEED)` with `cudnn.deterministic = True`. Multi-seed methods iterate over `SEEDS = [0, 1, 2, 3, 4]`.
- **CE-only baseline.** `kd/few_shot_kd.ipynb` reads transforms, optimizer, scheduler, loss config, etc. from the CE-only student's `metrics.json` — that folder must exist (for the chosen `STUDENT_NAME`) before launching KD.
