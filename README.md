## "Simplifying GNN Performance with Low Rank Kernel Models" Implementation
Dependencies: `torch`, `optuna`, `numpy`, `xarray`, `torch_geometric`, `tqdm`

For Table 2 results run:
`python low_rank_sweep.py --masks balanced --graphs none shift --save_name table2_results`

MLP2 results can be obtained by running `mlp_run.py`

Comparison of different evaluation is found in `eval_conventions.py`