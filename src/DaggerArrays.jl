module DaggerArrays

using Dagger: delayed, Thunk, Chunk, Context
using TensorOperations
using KernelAbstractions
using LoopVectorization
using Tullio

export delayed

export @dtullio

include("util.jl")

export Blocks, ArrayDomain
include("indices.jl")

export DArray, distribute
include("darray.jl")

include("tullio.jl")

include("arraylib.jl")

end # module
