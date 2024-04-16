## Quickstart


Set up your environment with:
```
conda init zsh
conda env create -f environment.yaml
conda activate in-context-learning
```


Run a training run specified by `<config_file>` with:
```
python src/ --config conf/train/<config_file>.yml
```

In order to create a hybrid architecture, create a copy of one of mod_transformers.py, mambaformers.py, etc. and then modify the instance variables for the block_var method and the logic of the forward_block method to your liking. Modify __init__.py under src/models, then modify conf/src/models.yml to adjust some of the finer model details (embed dim, no_attention, positional embeddings, etc.), and finally create your own training yml file for your custom model under conf/train.
