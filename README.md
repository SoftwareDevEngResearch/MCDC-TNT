# MCDC-TNT
Monte Carlo Deterministic Code - Transient Neutronics Testbed

To install clone then use command `pip install --user -e .` in project directory.

To run a simple intial initegration test run `python run.py -i tc_1.yaml` in mcdc_tnt.

To run a hardware test suit go to `tests/integration` and run `python test_hardware.py`

## Grading Notes:
1. **Installation:** This package only currenlty instals using local source files. It's reqirements for Numba CPU funcitonality are Numba, Numpy, Matplotlib, and Pyyaml
2. **Documentation:** A sphinx cite is linked in this git hub
3. **Testing:** I do note expect anyone to go throuhg the laborious task of setting up a pykokkos implementation to grade this work. As such I have removed the test files for it so that. This coupled with a lack of tests for the Numba GPU implementation makes my test coverage absmul. Please take this into consideration and that the numba CPU kernels and pure python kernels have decent test coverage
4. **Examples:** An example test suiet is listed to provide the same test case working accrose multiple pices of hardware
5. **License:** Is included in this directory
6. **Interface:** Runs with the commands provided in this README

# Instilation of PyKokkos
While most machines should be able to operate with the OpenMP backend currently on the Lassen Machine can get the CUDA version. To switch to the OpenMP only version change  `-DENABLE_CUDA` from `ON` to `OFF.

1. `git clone` [`pykokkos-base`](https://github.com/kokkos/pykokkos-base) and the develop granch of [`pykokkos`](github.com/kokkos/pykokkos). To do this in the pykokkos directory run `git fetch` then `git checkout develop`
2. Prep conda enviroment by snagging reqirements listed in reqirements.txt from pykokkos-base and pykokkos. (1. `conda create -n pyk` 2. `conda activate pyk` 3. in pyk-base directory run `conda install --file requirments.txt` 4. in pykokkos directory run `conda install --file requirments.txt`) *ensure that cmake is of version 18 or higher and that gcc/g++ versions are at least 9*
3. Install Pykokkos-base for both OpenMP, and CUDA implementations by running:
`python setup.py install -- -DCMAKE_CXX_COMPILER=g++ -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=ON -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON -G "Unix Makefiles" -- -j 4` *this will take upwards of 2 hours to build and will consume a considerable ammount of RAM*
4. Install pykokkos using `pip install --user -e .`
5. Run!

## Interface
This project is desigend to be interfaced with via the command line and an input.yaml file. An example is listed here:

```
name: 'fissioning_slab'   #name of the simluation (any string)
number of particles: 1e5  #number of particles top initiate in the 
rng seed: 777             #random number seed (int)
particle speed: 1         #particle speed (float)
neutrons per fission: 2   #how many neutrons to produce per fission event
isotropic: Ture           #isotropic source? if true than particles produced with a random direction

length of slab: 1         #width of the slab
surface locations: [0,1]  #region geometry deffitinition (vector of floats)

dx: 0.01   #mesh width (for error and scalar flux tracking) (float)

hardware target: nb_cpu          #specifying the hardware target: pp/nb_cpu/nb_gpu/pyk_cpu/pyk_gpu
print warmup times: True         #print warm up times

assemble mesh: True             #assemble mesh from crossections listed here
capture cross section: 0.333    #should be as many values here as regions specified in surface_locations
scatter cross section: 0.333
fission cross section: 0.333

file output: True      #should it output flux and stats? if a special file name is desiered supple in command line

error plot: True       #produce the error plot?
flux plot: True        #produce the flux plot?
```

Then to run a simulation it can be done from a python file using:
```
import mcdc_tnt
mcdc_tnt.run('input.yaml','output.out')
```

or from the command line in the mcdc_tnt directory with:
`python run.py -i input.yaml -o output.out`


## Acknowledgment
This work was supported by the Center for Exascale Monte-Carlo Neutron Transport (CEMeNT) a PSAAP-III project funded by the Department of Energy, grant number: DE-NA003967.
