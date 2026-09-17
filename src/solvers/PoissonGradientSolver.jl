include("PoissonSolver.jl")

"""
THEORY:
    The differential equation
        Δu = f,  where x ∈ Ω
    with boundary condition
        u(x) = bc(x) with x ∈ ∂Ω
    has solution u whose gradient at x is:
        ∇u(x) = 1/|B(x)| ∫_∂B u(y)n̂_y dy [boundary contribution]
                 - ∫_B f(y)∇G(x,y) dy    [source contribution]
    where B(x) is a ball centred at x
"""
function walkOnSphere_PoissonGradient(xᵢ::Point, ∂Ω::Scene, f::Function, WoS_depth::Integer, ϵ::Float64)
    WoS_depth ≥ 0 || throw(ArgumentError("WoS_depth must be non-negative"))
    r = distance(xᵢ, ∂Ω)

    n̂ = uniformDirectionSample(dim(xᵢ)) 
    dx = r * n̂
    xᵢ₊₁ = xᵢ ⊕ dx  # xᵢ₊₁ ~ Unif(∂B(xᵢ, r))
    boundary_contribution = walkOnSphere_Poisson(xᵢ₊₁, ∂Ω, f, WoS_depth, ϵ) ⊗ n̂ # ⊗ also act as scalar multiplication if u is a scalar field

    y = uniformBallPointSample(xᵢ, r) # y ~ Unif(B(xᵢ, r))
    source_contribution = -μ∂B(xᵢ, r) * f(y) ⊗ ∇G(xᵢ, y, r) # ⊗ also act as scalar multiplication if f is a scalar field

    return boundary_contribution + source_contribution
end

function solvePoissonGradient(x::Point, Ω::Scene, f::Function, WoS_depth::Integer, num_Samples::Integer, ϵ::Float64)
    num_Samples > 0 || throw(ArgumentError("num_Samples must be positive"))
    # Outputs are more accurate with high sample counts (≥500)
    return mean([walkOnSphere_PoissonGradient(x, Ω, f, WoS_depth, ϵ) for _ in 1:num_Samples])
end