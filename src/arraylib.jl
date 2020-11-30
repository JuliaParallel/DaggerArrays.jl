# map
using GeneralizedGenerated

function idxs(sym, T)
    ((Symbol(sym, i) for i=1:ndims(T))...,)
end

function Base.map(f, X::DArray)
    _map(f, X)
end

@gg function _map(f, X)
    i = idxs(:i, X)
    _dtullio(:(Z[$(i...)] := f(X[$(i...)])))
end
