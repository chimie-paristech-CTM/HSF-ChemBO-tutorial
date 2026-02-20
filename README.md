# HSF-ChemBO-tutorial

A tutorial on quick adoption of our proposed dimension-aware hyperprior for hidden-space representations in chemical Bayesian optimization.
The environment is based on [BayBE 0.12.2](https://emdgroup.github.io/baybe/0.12.2/) with Python 3.11.

For all the code and data used in the paper **Leveraging Hidden-Space Representations Effectively in Bayesian Optimization for Experiment Design through Dimension-Aware Hyperpriors**, see another [repo](https://github.com/chimie-paristech-CTM/HSF-ChemBO).


### Quickstart:

install git

download this repo: 
git clone https://github.com/chimie-paristech-CTM/HSF-ChemBO-tutorial


install [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

then run (all packages will be downloaded and saved locally)
```
uv run jupyter-lab
```
...or use vs code. Then execute
```
code .
```
and select the uv environment.
