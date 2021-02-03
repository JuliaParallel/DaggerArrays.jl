module DaggerArrays

import Dagger
using Dagger: delayed, istask, persist!, compute, Thunk, Chunk, Context
using RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

using Tullio

export delayed


include("util.jl")

export Blocks, ArrayDomain
include("indices.jl")

import Base.collect
export DArray, distribute
include("darray.jl")

export @dtullio
include("tullio.jl")

include("arraylib.jl")

end # module
