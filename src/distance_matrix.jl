struct FuncMatrix{T,F} <: AbstractMatrix{T}
    data::Matrix{T}
    f::F
end

Base.size(x::FuncMatrix) = (size(x.data, 2), size(x.data, 2))

Base.@propagate_inbounds function Base.getindex(x::FuncMatrix, i::Int, j::Int)
    return x.f(view(x.data, :, i), view(x.data, :, j))
end

function Base.sum(D::FuncMatrix)
    s = zeros(Threads.nthreads())
    Threads.@threads for i in 1:size(D, 2)
        tid = Threads.threadid()

        s[tid] += sum(D[:, i])
    end
    return sum(s)
end

