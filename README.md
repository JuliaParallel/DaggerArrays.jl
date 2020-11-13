## Experimental Distributed Arrays package

This is an attempt to use Tullio.jl to run tensor expressions on Dagger-based distributed arrays.

## Quick links

- You need [this branch](https://github.com/shashi/Tullio.jl/tree/s/lowered) of Tullio.jl
- [DArray struct](https://github.com/JuliaParallel/DaggerArrays.jl/blob/master/src/darray.jl#L15-L24) is how we represent a distributed array
- [`@dtullio` macro](https://github.com/JuliaParallel/DaggerArrays.jl/blob/master/src/tullio.jl#L127-L128) is in this file


## What works

```julia
julia> using DaggerArrays

julia> A = distribute(reshape(1:24, (4,6)), Blocks(2,3))
4×6 DArray{Int64, 2} with 2×2 chunks

julia> B = distribute(reshape(1:24, (6,4)), Blocks(3,2))
6×4 DArray{Int64, 2} with 2×2 chunks

julia> @dtullio C[i,k] := A[i,j] * B[j,k]
4×4 DArray{Any, 2} with 2×2 chunks
```

## What does not work (and should)


```julia
julia> @generated matmul(A, B) = :(@dtullio C[i,k] := A[i,j] * B[j,k])
matmul (generic function with 1 method)

julia> matmul(A,B)
ERROR: The function body AST defined by this @generated function is not pure. This likely means it contains a closure or comprehension.
Stacktrace:
 [1] top-level scope at REPL[24]:1
 [2] run_repl(::REPL.AbstractREPL, ::Any) at /build/julia/src/julia-1.5.2/usr/share/julia/stdlib/v1.5/REPL/src/REPL.jl:288
```

This is required to implement dimension-independent operations

```julia
julia> A = distribute(reshape(1:24, (4,6)), Blocks(2,3))
4×6 DArray{Int64, 2} with 2×2 chunks

julia> B = distribute(reshape(1:24, (6,4)), Blocks(6,2))
6×4 DArray{Int64, 2} with 1×2 chunks

julia> @dtullio C[i,k] := A[i,j] * B[j,k]
ERROR: range of index j must agree
Stacktrace:
 [1] top-level scope at /home/shashi/.julia/dev/Tullio/src/macro.jl:777
 [2] top-level scope at /home/shashi/.julia/dev/DaggerArrays/src/tullio.jl:103
 [3] run_repl(::REPL.AbstractREPL, ::Any) at /build/julia/src/julia-1.5.2/usr/share/julia/stdlib/v1.5/REPL/src/REPL.jl:288
```
(Should auto-slice B to match the layout of A)
