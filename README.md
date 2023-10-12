## "Simplifying GNN Performance with Low Rank Kernel Models" Implementation
Paper link: [https://arxiv.org/abs/2310.05250](https://arxiv.org/abs/2310.05250)

Dependencies: `torch`, `optuna`, `numpy`, `xarray`, `torch_geometric`, `tqdm`

For Table 2 results run:
`python low_rank_sweep.py --masks balanced --graphs none shift --save_name table2_results`

MLP2 results can be obtained by running `mlp_run.py`

Comparison of different evaluation is run using `eval_conventions.py`