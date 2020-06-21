1. `01_navier_stokes.py`: Port of `13_pde/10_cavity.ipynb` from Jupyter notebook to Python script.
    - Stopping point is changed from number of steps to error threshold (`eps`), which signifies convergence.
    - Outputs number of steps to convergence, elapsed time, and the sums of the absolute values of `u`, `v`, and `p` for quick sanity check.
    - Exports figure as `fig_python.png`.
2. `02_navier_stokes.cpp`: Port of `01_navier_stokes.py` to C++.
    - Slower than Python version, due to Numpy operations ported naively.
    - Results (`u`, `v`, `p`) are saved as `02_navier_stokes.out`.
    - `plot.py`: Plots Navier-Stokes calculation results
    - Exports figure as `fig_cpp.png` with `plot.py`.
3. `03_cuda.cpp`: CUDA parallelization of `02_navier_stokes.cpp`.
    - Some CUDA functions cannot accept `double` variable types, thus variables are defined as `float`s. This produces slight differences in output.
    - Results (`u`, `v`, `p`) are saved as `03_cuda.out`.
    - Exports figure as `fig_cuda.png` with `plot.py`.
4. `04_openmp.cpp`: OpenMP parallelization of `02_navier_stokes.cpp`.
    - Results (`u`, `v`, `p`) are saved as `04_openmp.out`.
    - Exports figure as `fig_openmp.png` with `plot.py`.
5. `05_openacc.cpp`: OpenACC version of `02_navier_stokes.cpp`, directly ported from `04_openmp.cpp`.
    - Not tested yet, will test on Tsubame
    - Results (`u`, `v`, `p`) are saved as `05_openacc.out`.
    - Exports figure as `fig_openacc.png` with `plot.py`.

Results on local machine, with Intel i5-7600, 8GB of RAM, and NVIDIA TITAN Xp, running Ubuntu 18.04

```console
foo@bar:~$ python 01_navier_stokes.py 
Steps:  6605
Elapsed time:  8.2008 s.
Sum(|u|)= 219.63547247647466
Sum(|v|)= 129.10805866433103
Sum(|p|)= 175.6803115004912
Figure exported as fig_python.png

foo@bar:~$ g++ 02_navier_stokes.cpp && ./a.out && python plot.py 02_navier_stokes.out fig_cpp.png
Steps: 6605
Elapsed time: 6.461589 s.
Sum(|u|)=219.635472
Sum(|v|)=129.108059
Sum(|p|)=175.680312
Figure exported as fig_cpp.png

foo@bar:~$ nvcc 03_cuda.cu && ./a.out && python plot.py 02_navier_stokes.out fig_cuda.png
Steps: 6595
Elapsed time: 3.871475 s.
Sum(|u|)=219.635117
Sum(|v|)=129.107422
Sum(|p|)=175.680099
Figure exported as fig_cuda.png

foo@bar:~$ g++ -fopenmp 04_openmp.cpp && ./a.out && python plot.py 04_openmp.out fig_openmp.png
Steps: 6605
Elapsed time: 3.524046 s.
Sum(|u|)=219.635472
Sum(|v|)=129.108059
Sum(|p|)=175.680312
Figure exported as fig_openmp.png
```