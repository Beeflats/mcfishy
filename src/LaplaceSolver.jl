include("BoundaryCondition.jl")

mean(x) = sum(x) / length(x)

"""
Solve the equation
    Δu = 0,  where x ∈ Ω
with boundary condition
    u(x) = bc(x) with x ∈ ∂Ω
"""

function walkOnSphere(xᵢ::Point, ∂Ω::Scene, WoS_depth::Integer,ϵ ::Real)
    WoS_depth ≥ 0 || throw(ArgumentError("WoS_depth must be non-negative"))
    dx = distance(xᵢ, ∂Ω)
    if WoS_depth == 0 || dx ≤ ϵ
        x_nearest, obj = nearest(xᵢ, ∂Ω)
        return obj(x_nearest)
    else
        ω = uniformSampleCircle()
        xᵢ₊₁ = xᵢ ⊕ dx * ω
        return walkOnSphere(xᵢ₊₁, ∂Ω, WoS_depth - 1, ϵ)
    end
end

function solveLaplace(x::Point, Ω::Scene, WoS_depth::Integer, num_Samples::Integer, ϵ::Real)
    num_Samples > 0 || throw(ArgumentError("num_Samples must be positive"))
    return mean([walkOnSphere(x, Ω, WoS_depth, ϵ) for _ in 1:num_Samples])
end

