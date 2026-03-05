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
You can get the audio datasets from [GoogleDrive](https://drive.google.com/file/d/1FnpYhMaeskckxGheKjar0U4YHIdDKM6K/view). Please extract the data under the `./data/{dataset_name}/mp3/`

## How to run VGRec

### MFE

- Coat

```
python run_MFE.py --dataset=coat --epochs=35 --lr=0.0005 --test=1 --neg_num=3 
```

- Movielens-1m

```
python run_MFE.py --dataset=movielens1m --epochs=35 --lr=0.0005 --test=1 --neg_num=5 
```

### AGIP

- Coat

```
python main.py --weight_decay=1e-4 --lr=0.001 --n_layers=3 --dataset=coat --recdim=64 --neg_num 3 --ens_ratio 0.8
```

- Movielens-1m

```
python main.py --weight_decay=1e-4 --lr=0.001 --n_layers=3 --dataset=movielens1m --recdim=64 --neg_num 5 --ens_ratio 0.6
```



## Benchmarking

Coat:
|   Metrics   | VERS |
| ----------- | ----------- |
|  F1@10   |    **12.68**   |
|  Precision@10   |    **8.24**   |
| Recall@10   |    **27.47**   |
|  NDCG@10    |    **18.68**   |

[//]: # (## Citation)

[//]: # ()
[//]: # (You are welcome to cite our paper:)

[//]: # (```)

[//]: # ()
[//]: # (```)
