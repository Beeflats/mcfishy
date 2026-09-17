include("./SolverUtils.jl")

"""
Solve the equation
    Δu = 0,  where x ∈ Ω
with boundary condition
    u(x) = bc(x) with x ∈ ∂Ω
"""

function walkOnSphere(xᵢ::Point, ∂Ω::Scene, WoS_depth::Integer, ϵ::Float64)
    WoS_depth ≥ 0 || throw(ArgumentError("WoS_depth must be non-negative"))
    r = distance(xᵢ, ∂Ω)
    if WoS_depth == 0 || r ≤ ϵ
        x, bc = nearest(xᵢ, ∂Ω)
        return bc(x)
    else
        dx = r * uniformDirectionSample(dim(xᵢ))
        xᵢ₊₁ = xᵢ ⊕ dx # xᵢ₊₁ ~ Unif(∂B(xᵢ, 1))
        return walkOnSphere(xᵢ₊₁, ∂Ω, WoS_depth - 1, ϵ)
    end
end

function solveLaplace(x::Point, Ω::Scene, WoS_depth::Integer, num_Samples::Integer, ϵ::Float64)
    num_Samples > 0 || throw(ArgumentError("num_Samples must be positive"))
    return mean([walkOnSphere(x, Ω, WoS_depth, ϵ) for _ in 1:num_Samples])
end