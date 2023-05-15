## Mundy GPU

### May 09

+ The sample UnitTestMundy runs on the GPU, and converges.
+ Performance figures:

```
Np,     OpenFPM, Stk,   This (GPU), This (CPU), Speedup vs Best  
428750, .538,   .362,    .146,                      2.48x
640000, .842,   .553,    .369,        18.23         1.49x
911250, 1.246,  .973.    .286,                      3.4x
1250000, 2.178, 1.508,   ---, ---

```
+ UnitTests: Integration with Gtest complete, tested w/ some samples for host & device codes.

### Comments:
+ It fails to converge for (around) >1 million particles.
+ 814 MB peak device memory usage for Np=640,000. Tested on Amd20 with V100 nodes.
+ Stk: have to manually call sync for every field modified. Initially biggest source of bugs.

### ToDO
1. Eigen Integration.
2. Write unit tests for kernels.
3. Fix: Bug somewhere preventing integration of ArborX's Neighbor search to current code
4. Investigate & find why crashes for ~ 1 million and above particles.
5. Rebase with MundyScratch `main` branch. 
6. Profile and think about performance.


<br>


### May 15

+ Integration with Eigen.
    + In GPU, componenets of a field are not laid out contigiuosly. So can't map to Eigen type without copying.
    + E.g.: Position. x,y and z components are not contigious like host (on device: *y = *(x+bucket_capacity)
    + Still, replaced basic algebra like dot/ cross product with Eigen.

+ Why doesn't it work for large number of particles? 
    + Made the codebase general enough so it runs on both host & device (which it always should).
    + Ran the code on Host. [1]
        + Found no convergence issues! Ran perfectly well for more than 2 million particles
        + Serial code. (Couldn't get OpenMP only Trilinos/Kokkos to work for some reason, so no profiling against Bryce's implementation yet).

    + Noticed the solver "sometimes" works. Below is the performance results from them: 

```
Np,     OpenFPM-GPU, Stk,   This (GPU), Speedup vs Best  
428750, .538,   .362,    .146,       2.48x
640000, .842,   .553,    .369,       1.49x
911250, 1.246,  .973.    .286,       3.4x
1250000, 2.178, 1.508,   .410,       3.65x
1663750, 2.72,  3.113,   .624,       4.36x
2160000, 3.52,  4.101,   .783,       4.61x

```

+ Setup profiling using `Kokkos-tools`.
:
    + Basic usage example, showing top 5 kernels. More targeted profiling will be necessary.

```
Name, Total Time, Calls, Time/call, %of Kokkos Time, %of Total Time

- Kokkos::View::destruction []
 (ParFor)   0.498339 34416 0.000014 47.358629 3.075142
- stk::mesh::TeamFunctor "compute_constraint_center_of_mass_force_torque"
 (ParFor)   0.253949 67 0.003790 24.133578 1.567068
- transpose_from_pinned_and_mapped_memory
 (ParFor)   0.099228 35 0.002835 9.429927 0.612314
- stk::mesh::TeamFunctor "compute_rate_of_change_of_sep"
 (ParFor)   0.087599 67 0.001307 8.324801 0.540555
- transpose_to_zero_copy_pinned_memory
 (ParFor)   0.067234 9 0.007470 6.389431 0.414886
........

Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                  16.20539 seconds
Total Time in Kokkos kernels:                                       1.05227 seconds
   -> Time outside Kokkos kernels:                                 15.15312 seconds
   -> Percentage in Kokkos kernels:                                    6.49 %
Total Calls to Kokkos Kernels:                                        34971

-------------------------------------------------------------------------

```



### ToDO
+ Integrate with MundyScratch? 
+ Complete Eigen Integration
+ Fix: Nearest neighbor bug
+ More unit tests for device code.

[1] To compile Trilinos, had to turn off STK_tests. It was using nvcc for host code for one partciular test file. Haven't found the root cause. 
