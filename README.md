# MCDC-TNT
Monte Carlo Deterministic Code - Transient Neutronics Testbed

To install use command `pip install mcdc_tnt`

To run a simple intial initegration test run `python run.py -i tc_1.yaml`.

To run a hardware test suit go to `tests/integration` and run `python test_hardware.py`

# Instilation of PyKokkos
While most machines should be able to operate with the OpenMP backend currently on the Lassen Machine can get the CUDA version. To switch to the OpenMP only version change  `-DENABLE_CUDA` from `ON` to `OFF.

1. `git clone` [`pykokkos-base`](https://github.com/kokkos/pykokkos-base) and the develop granch of [`pykokkos`](github.com/kokkos/pykokkos). To do this in the pykokkos directory run `git fetch` then `git checkout develop`
2. Prep conda enviroment by snagging reqirements listed in reqirements.txt from pykokkos-base and pykokkos. (1. `conda create -n pyk` 2. `conda activate pyk` 3. in pyk-base directory run `conda install --file requirments.txt` 4. in pykokkos directory run `conda install --file requirments.txt`) *ensure that cmake is of version 18 or higher and that gcc/g++ versions are at least 9*
3. Install Pykokkos-base for both OpenMP, and CUDA implementations by running:
`python setup.py install -- -DCMAKE_CXX_COMPILER=g++ -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=ON -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON -G "Unix Makefiles" -- -j 4` *this will take upwards of 2 hours to build and will consume a considerable ammount of RAM*
4. Install pykokkos using `pip install --user -e .`
5. Run!


This work was supported by the Center for Exascale Monte-Carlo Neutron Transport (CEMeNT) a PSAAP-III project funded by the Department of Energy, grant number: DE-NA003967.
