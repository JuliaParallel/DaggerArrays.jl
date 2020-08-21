module DaggerArrays

import Dagger
using Dagger: delayed, istask, persist!, compute, Thunk, Chunk, Context

using TensorOperations
using KernelAbstractions
using LoopVectorization
using Tullio

export delayed

export @dtullio

include("util.jl")

export Blocks, ArrayDomain
include("indices.jl")

import Base.collect
export DArray, distribute
include("darray.jl")

include("tullio.jl")

include("arraylib.jl")

end # module
