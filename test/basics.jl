using Test, DaggerArrays

using Random

Random.seed!(123)

function get_rand_chunker(sz)
    map(sz) do s
        rand(1:(s+2))
    end |> Blocks
end

function get_rand_slice(sz)
    map(sz) do s
        if rand() < 0.05
            Colon()
        #elseif rand() < 0.05
        #    s:s+1
        elseif rand() < 0.1
            rand(1:s)
        else
            i = rand(1:s)
            j = rand(i-1:s)
            i:j
        end
    end
end

function make_some_array(sz, ch = get_rand_chunker(sz), T = rand([Float64, Int]))
    reshapecont(T, sz) = T.(reshape(1:prod(sz), sz))
    f = rand([rand, zeros, ones, reshapecont])
    A = f(T, sz)
    distribute(A, ch), A
end


@testset "distribute" begin
    for i=1:3000
        dA, A = make_some_array((rand(1:10), rand(1:10)))
        @test collect(dA) == A
    end
end

@testset "indexing" begin
    for i=1:3000
        dA, A = make_some_array((rand(1:10), rand(1:10)))

        idx = get_rand_slice(size(A))

        dX = collect(dA[idx...])
        X = collect(A[idx...])

        if dX != X
            println("getindex failed: $(dA) vs $(summary(A)) with index $(idx)")
            @test_skip false
        else
            @test true
        end
    end
end

distribute(rand(5,5), Blocks(6, 2))
