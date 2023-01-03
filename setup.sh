#!/usr/bin/env sh
set -o errexit
set -o nounset
set -o pipefail


# conda create -n sbibm-smnle-py39 python=3.9

if [ "$(uname)" = "Darwin" ]; then
    echo "macos"


    conda install -y -c conda-forge mpi4py openmpi openblas
    conda install -y -c conda-forge cmake=3.19 eigen ninja
    pip install cmaketools
    conda install -y -c conda-forge scipy=1.9.3 numpy=1.20.3 scikit-learn matplotlib pandas seaborn
    pip install --no-use-pep517 git+https://github.com/diyabc/abcranger



    conda install -y pytorch -c pytorch
    conda install -y -c conda-forge pyro-ppl

    pip install ipython pyfzf
    pip install --upgrade "jax[cpu]" flax chex numpyro

    conda install -c conda-forge -y grpcio
    pip install  sbi sbibm

    # install scipy==1.7.3 after having isnstalled sbi (otherwise sbi would
    # force reinstallation of scipy>= 1.9.3, making SM-ExpFam-LFI fail)
    conda install -y scipy=1.7.3

    pip install  pymc3
    pip install blackjax
    pip install abcpy

elif [ "$(uname)" = "Linux" ]; then
    echo "linux"
    conda install -y -c conda-forge mpi4py openblas
    conda install -y -c conda-forge cmake=3.19 eigen ninja
    pip install cmaketools

    conda install -y -c conda-forge scipy=1.9.3 numpy=1.20.3 scikit-learn matplotlib pandas seaborn
    pip install pyabcranger
    conda install -y -c conda-forge scipy numpy scikit-learn matplotlib pandas seaborn

    conda install -y pytorch -c pytorch
    conda install -y -c conda-forge pyro-ppl

    pip install ipython pyfzf
    pip install --upgrade "jax[cpu]" flax chex numpyro

    pip install sbi sbibm
    pip install blackjax
    pip install abcpy

    conda install pymc3

    # install scipy==1.7.3 after having isnstalled sbi (otherwise sbi would
    # force reinstallation of scipy>= 1.9.3, making SM-ExpFam-LFI fail)
    conda install -y scipy=1.7.3

fi

pip install -e ./vendor/sbi_ebm
