# GPAWP
Heterogeneous Graph Prompt Learning via Adaptive Weight Pruning
## Abstract
Graph Neural Networks (GNNs) have achieved remarkable success in various graph-based tasks (e.g., node
classification or link prediction). Despite their triumphs, GNNs still face challenges such as long training
and inference times, difficulty in capturing complex relationships, and insufficient feature extraction. To
tackle these issues, graph pre-training and graph prompt methods have garnered increasing attention for
their ability to leverage large-scale datasets for initial learning and task-specific adaptation, offering potential
improvements in GNN performance. However, previous research has overlooked the potential of graph
prompts in optimizing models, as well as the impact of both positive and negative graph prompts on model
stability and efficiency. To bridge this gap, we propose GPAWP, a novel framework that integrates adaptive
weight pruning into graph prompt learning. Unlike prior approaches that indiscriminately retain all prompts,
GPAWP dynamically evaluates and prunes graph prompts based on their significance, effectively reducing
redundancy while improving efficiency. The framework comprises three core modules: (A) Tuning, which
ensures effective feature extraction and embedding refinement; (B) Evaluation & Pruning, which identifies and
removes less important graph prompts; and (C) Retuning, which optimizes the retained prompts to maximize
downstream performance. Extensive experiments on three benchmark datasets demonstrate the superiority
of GPAWP, leading to a significant reduction in parameters in node classification tasks.
## Requirements

```
conda create -n GPAWP python=3.8
conda activate GPAWP
pip install -r requirements.txt
```

## Datasets
You can view the datasets used directly in the data folder.

## How to run GPAWP
The default dataset is ACM. You need to change the corresponding parameters in pre_train.py, run.py, and retrain.py to train and evaluate on other datasets.

### Pretrain

```
python pre_train.py

```

### Tuning

```
python run.py
```
### Retuning

```
python retrain.py
```

