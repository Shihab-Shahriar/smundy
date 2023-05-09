## Mundy GPU

+ The sample UnitTestMundy runs on the GPU, and converges.
+ Performance figures:

```
Np,     OpenFPM, Stk,   This, Speedup vs Best  
428750, .538,   .362,   .146, 2.48x
640000, .842,   .553,   .369, 1.49x
911250, 1.246,  .973.   .286, 3.4x
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



