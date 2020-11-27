# map

using GeneralizedGenerated

function idxs(sym, T)
    ((Symbol(sym, i) for i=1:ndims(T))...,)
end

@generated function Base.map(f, X::DArray)
    i = idxs(:i, X)
    @eval @dtullio Z[$(i...)] := f(X[$(i...)])
end
