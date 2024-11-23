# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

---

# 3.1 and 3.2 Diagnostics
<details>
<summary>Click to expand</summary>

```
(.venv) C:\Users\Owner\MLEworkspace\mod3-kpan02>python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (164)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        strides_match = np.array_equal(in_strides, out_strides)              |
        size = int(np.prod(out_shape))---------------------------------------| #2
                                                                             |
        for i in prange(size):-----------------------------------------------| #3
            if strides_match:                                                |
                out[i] = fn(in_storage[i])                                   |
                                                                             |
            else:                                                            |
                out_index = np.zeros(MAX_DIMS, np.int32)---------------------| #0
                in_index = np.zeros(MAX_DIMS, np.int32)----------------------| #1
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                o = index_to_position(out_index, out_strides)                |
                j = index_to_position(in_index, in_strides)                  |
                out[o] = fn(in_storage[j])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (180) is hoisted
out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (181) is hoisted
out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (214)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (214)
--------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                         |
        out: Storage,                                                                 |
        out_shape: Shape,                                                             |
        out_strides: Strides,                                                         |
        a_storage: Storage,                                                           |
        a_shape: Shape,                                                               |
        a_strides: Strides,                                                           |
        b_storage: Storage,                                                           |
        b_shape: Shape,                                                               |
        b_strides: Strides,                                                           |
    ) -> None:                                                                        |
        strides_match = np.array_equal(a_strides, out_strides) and np.array_equal(    |
            b_strides, out_strides                                                    |
        )                                                                             |
        shape_match = np.array_equal(a_shape, b_shape)                                |
        size = int(np.prod(out_shape))------------------------------------------------| #7
                                                                                      |
        for i in prange(size):--------------------------------------------------------| #8
            if strides_match and shape_match:                                         |
                out[i] = fn(a_storage[i], b_storage[i])                               |
            else:                                                                     |
                out_index = np.zeros(MAX_DIMS, np.int32)------------------------------| #4
                a_pos = np.zeros(MAX_DIMS, np.int32)----------------------------------| #5
                b_pos = np.zeros(MAX_DIMS, np.int32)----------------------------------| #6
                to_index(i, out_shape, out_index)                                     |
                o = index_to_position(out_index, out_strides)                         |
                broadcast_index(out_index, out_shape, a_shape, a_pos)                 |
                j = index_to_position(a_pos, a_strides)                               |
                broadcast_index(out_index, out_shape, b_shape, b_pos)                 |
                k = index_to_position(b_pos, b_strides)                               |
                out[o] = fn(a_storage[j], b_storage[k])                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #7, #8).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)



Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (235) is hoisted
out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (236) is hoisted
out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_pos = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (237) is hoisted
out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_pos = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (270)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (270)
---------------------------------------------------------------|loop #ID
    def _reduce(                                               |
        out: Storage,                                          |
        out_shape: Shape,                                      |
        out_strides: Strides,                                  |
        a_storage: Storage,                                    |
        a_shape: Shape,                                        |
        a_strides: Strides,                                    |
        reduce_dim: int,                                       |
    ) -> None:                                                 |
        size = int(np.prod(out_shape))-------------------------| #10
        reduce_size = a_shape[reduce_dim]                      |
                                                               |
        for i in prange(size):---------------------------------| #12
            out_index: Index = np.zeros(MAX_DIMS, np.int32)----| #9
            to_index(i, out_shape, out_index)                  |
            o = index_to_position(out_index, out_strides)      |
            for j in prange(reduce_size):----------------------| #11
                out_index[reduce_dim] = j                      |
                p = index_to_position(out_index, a_strides)    |
                out[o] = fn(out[o], a_storage[p])              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #10, #12, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--12 is a parallel loop
   +--9 --> rewritten as a serial loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--9 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--9 (serial)
   +--11 (serial)



Parallel region 0 (loop #12) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#12).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (283) is hoisted
out of the parallel loop labelled #12 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (294)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\Owner\MLEworkspace\mod3-kpan02\minitorch\fast_ops.py (294)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
    for n in prange(out_shape[0]):--------------------------------------------------------| #13
        for i in range(out_shape[1]):  # Rows of a                                        |
            for j in range(out_shape[2]):  # Columns of b                                 |
                sum = 0                                                                   |
                a_pos = n * a_batch_stride + i * a_strides[1]                             |
                b_pos = n * b_batch_stride + j * b_strides[2]                             |
                out_pos = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]    |
                                                                                          |
                for k in range(a_shape[-1]):  # Columns of a and rows of b                |
                    sum += a_storage[a_pos] * b_storage[b_pos]                            |
                    a_pos += a_strides[2]                                                 |
                    b_pos += b_strides[1]                                                 |
                                                                                          |
                out[out_pos] = sum                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #13).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
</details>
