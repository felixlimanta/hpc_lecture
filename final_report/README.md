1. `01_navier_stokes.py`: Port of `13_pde/10_cavity.ipynb` from Jupyter notebook to Python script.
    - Stopping point is changed from number of steps to error threshold (`eps`), which signifies convergence.
    - Outputs number of steps to convergence, elapsed time, and the sums of the absolute values of `u`, `v`, and `p` for quick sanity check.
    - Exports figure as `fig_python.png`.
2. `02_navier_stokes.cpp`: Port of `01_navier_stokes.py` to C++.
    - Results (`u`, `v`, `p`) are saved as `02_navier_stokes.out`.
    - `plot.py`: Plots Navier-Stokes calculation results


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
Elapsed time: 18.985322 s.
Sum(|u|)=219.635472
Sum(|v|)=129.108059
Sum(|p|)=175.680312
Figure exported as fig_cpp.png
```