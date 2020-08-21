module DaggerArrays

using Dagger: delayed, chunks, Thunk, Chunk
using TensorOperations
using KernelAbstractions
using LoopVectorization
using Tullio

export delayed

export @dtullio

export DArray, Blocks, ArrayDomain
include("indices.jl")

include("darray.jl")

include("tullio.jl")

include("arraylib.jl")

end # module
