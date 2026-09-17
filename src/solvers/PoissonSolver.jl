include("./SolverUtils.jl")

"""
Solve the equation
    Δu = f,  where x ∈ Ω
with boundary condition
    u(x) = bc(x) with x ∈ ∂Ω
"""

function walkOnSphere_Poisson(xᵢ::Point, ∂Ω::Scene, f::Function, WoS_depth::Integer, ϵ::Float64)
    WoS_depth ≥ 0 || throw(ArgumentError("WoS_depth must be non-negative"))
    r = distance(xᵢ, ∂Ω)
    if WoS_depth == 0 || r ≤ ϵ
        x, bc = nearest(xᵢ, ∂Ω)
        return bc(x)
    else
        dx = r * uniformDirectionSample(dim(xᵢ)) 
        xᵢ₊₁ = xᵢ ⊕ dx # xᵢ₊₁ ~ Unif(∂B(xᵢ, r))
        boundary_contribution = walkOnSphere_Poisson(xᵢ₊₁, ∂Ω, f, WoS_depth - 1, ϵ)

        y = uniformBallPointSample(xᵢ, r) # y ~ Unif(B(xᵢ, r))
        source_contribution = -μ∂B(xᵢ, r) * f(y) * G(xᵢ, y, r)
    
        return boundary_contribution + source_contribution
    end
end

function solvePoisson(x::Point, Ω::Scene, f::Function, WoS_depth::Integer, num_Samples::Integer, ϵ::Float64)
    num_Samples > 0 || throw(ArgumentError("num_Samples must be positive"))
    return mean([walkOnSphere_Poisson(x, Ω, f, WoS_depth, ϵ) for _ in 1:num_Samples])
end
