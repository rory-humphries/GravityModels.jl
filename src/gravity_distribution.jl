struct GravityDistributionData{OF,IF,D}
    outflow::OF
    inflow::IF
    dist::D
end

# investigate possibility of outflow, inflow being matrices and dist being a tensor
# to allow multiple input variables i.e.
# A(of[1], of[2], ...)B(if[1], if[2], ...)/F(dij[1], dij[2], ...)
struct GravityDistribution{GM,OF,IF,D}
    gm::GM

    data::GravityDistributionData{OF,IF,D}

    C::Float64

    function GravityDistribution{GM,OF,IF,D}(
        gm::GM, data::GravityDistributionData{OF,IF,D}
    ) where {GM,OF,IF,D}
        @unpack outflow, inflow, dist = data

        Cvec = zeros(Threads.nthreads())

        Threads.@threads for i in 1:length(outflow)
            tid = Threads.threadid()
            Cvec[tid] += sum(gm.(outflow[i], inflow, dist[:, i]))
        end
        C = sum(Cvec)
        return new(gm, data, C)
    end
end

function GravityDistribution(
    gm::GM, data::GravityDistributionData{OF,IF,D}
) where {GM,OF,IF,D}
    return GravityDistribution{GM,OF,IF,D}(gm, data)
end

function Distributions.pdf(gd::GravityDistribution, x::AbstractVector)
    return gd.gm(gd.data.outflow[x[1]], gd.data.inflow[x[2]], gd.data.dist[x[1], x[2]]) /
           gd.C
end

function Distributions.pdf(gd::GravityDistribution, x::AbstractMatrix)
    return map(i -> pdf(gd, view(x, :, i)), axes(x, 2))
end

function Distributions.sample(gd::GravityDistribution, N)
    @unpack outflow, inflow, dist = gd.data

    dims = size(gd.data.dist, 2)
    pout = ones(dims)
    Threads.@threads for i in 1:length(pout)
        pout[i] = sum(gd.gm.(outflow[i], inflow, dist[:, i]))
    end

    samps = StatsBase.sample(1:dims, weights(pout), N)

    isamples = Int64[]
    jsamples = Int64[]
    sizehint!(isamples, N)
    sizehint!(jsamples, N)
    for (i, n) in countmap(samps)
        p = gd.gm.(outflow[i], inflow, dist[:, i])
        w = Weights(p)
        j = StatsBase.sample(1:length(w), w, n)

        append!(isamples, repeat([i], n))
        append!(jsamples, j)
    end

    return [isamples jsamples]
end
