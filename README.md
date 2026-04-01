# A VQ-VAE + Transformer workflow for FCC-ee tagger
The overall strategy is to learn a task-agnostic, physics-informed discrete representation of jet constituents via unsupervised tokenization, which can then be reused as a stable and interpretable input for Transformer-based jet classification at FCC-ee.
Jets are represented as sets of particle-flow constituents and encoded using a learned discrete vocabulary obtained via vector quantization. An unsupervised VQ-VAE tokenizer is trained to map constituent-level observables into a compact set of tokens that summarize recurring particle-level patterns while remaining agnostic to the downstream task. This representation reduces the dimensionality and variability of jet inputs and provides a natural interface to Transformer-based models, enabling efficient pre-training and systematic studies of generalization across jet energies and processes.

# bash commands
```
source /cvmfs/sw.hsf.org/key4hep/releases/2023-06-05-fcchh/x86_64-centos7-gcc12.2.0-opt/key4hep-stack/*/setup.sh
ulimit -s unlimited
python3 process_data.py
singularity shell -B /gpfs01/ --nv /usatlas/u/asciandra/colorsinglet.sif
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python3 training.py
```
