# NBIQ Dataset
We construct a large and difficult datasets called NBIQ to test our algorithms, since existing public available datasets are generally easy for modern solvers and computing devices, and the results on them are hard to tell which algorithm is better. 

## Generate NBIQ instance
The python file nbiq_gen.py gives a function to generate the NBIQ instances.
```
# nvar: number of variables
# neg_ratio: ratio of the negative weights
# dens_ratio: density of the matrix
data = gen_data(nvar, neg_ratio, dens_ratio)
```
Since the full datasets we have shown in the paper are too large for downloading, we give the generating codes with a fixed numpy random seed. To generate the complete datasets, run the following code
```
python nbiq_gen.py
```
and then check the consistency of the generating data files with the given 3 nbiq data with 5000 variables.