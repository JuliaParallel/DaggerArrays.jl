# map

using GeneralizedGenerated

function idxs(sym, T)
    ((Symbol(sym, i) for i=1:ndims(T))...,)
end

@gg function Base.map(f, X::DArray, Y::AbstractArray...)
    i = idxs(:i, X)
    :(@dtullio Z[$(i...)] := f(X[$(i...)]))
end
