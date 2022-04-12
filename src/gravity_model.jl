abstract type AbstractGravityModel end

# ========================================
# Exponential
# ========================================

struct GravityExp <: AbstractGravityModel
    α::Float64
    β::Float64
    γ::Float64
end

function (gm::GravityExp)(mi, mj, d)
    @unpack α, β, γ = gm
    if d == 0 || mi == 0 || mj == 0
        return 0.0
    else
        return (mi^α) * (mj^β) * exp(d * γ)
    end
end

# ========================================
# Power law
# ========================================

struct GravityPl <: AbstractGravityModel
    α::Float64
    β::Float64
    γ::Float64
end

function (gm::GravityPl)(mi, mj, d)
    @unpack α, β, γ = gm
    if d == 0 || mi == 0 || mj == 0
        return 0.0
    else
        return (mi^α) * (mj^β) * (d^γ)
    end
end

# ========================================
# Exponential and power law
# ========================================

struct GravityExpPl <: AbstractGravityModel
    α::Float64
    β::Float64
    γ::Float64
    κ::Float64
end

function (gm::GravityExpPl)(mi, mj, d)
    @unpack α, β, γ, κ = gm
    if d == 0 || mi == 0 || mj == 0
        return 0.0
    else
        return (mi^α) * (mj^β) * (d^γ) * exp(d * κ)
    end
end

# ========================================
# Log normal
# ========================================

struct GravityLogNormal <: AbstractGravityModel
    α::Float64
    β::Float64
    μ::Float64
    σ::Float64
end

function (gm::GravityLogNormal)(mi, mj, d)
    @unpack α, β, μ, σ = gm
    if d == 0 || mi == 0 || mj == 0
        return 0.0
    else
        return (mi^α) * (mj^β) * exp(-(log(d) - μ)^2 / (2 * σ^2)) / (d * σ * sqrt(2 * π))
    end
end

