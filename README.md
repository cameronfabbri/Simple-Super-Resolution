# Simple-Super-Resolution
Minimal implementation of a Super Resolution module using GANs

# Environment
`conda env create -f environment.yml`

# To train
`python -m scripts.main --data_dir=data/`

On a mac:
`PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.run`

Images must exist in `data/train` and `data/test`
