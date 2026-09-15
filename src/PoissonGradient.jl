include("BoundaryCondition.jl")
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

mean(x) = sum(x) / length(x)

function uniformSampleDisk(center::Point, radius::Real)
    r = radius * sqrt(rand())
    ω = uniformSampleCircle()
    return center ⊕ r * ω
end

G(x, y, dx) = 1/(2π) * log(dx / distance(x,y))
∇G(x, y, dx) = (x → y)/(2π) * (1/distance(x,y)^2 - 1/dx^2)

function walkOnSphere_PoissonGradient(xᵢ::Point, ∂Ω::Scene, f::Function, WoS_depth::Integer,ϵ ::Real)
    WoS_depth ≥ 0 || throw(ArgumentError("WoS_depth must be non-negative"))
    dx = distance(xᵢ, ∂Ω)

    n̂ = uniformSampleCircle()
    xᵢ₊₁ = xᵢ ⊕ dx * n̂
    boundary_contribution = walkOnSphere_Poisson(xᵢ₊₁, ∂Ω, f, WoS_depth, ϵ) * n̂

    υ = uniformSampleCircle()
    r = dx * sqrt(rand())
    y = xᵢ ⊕ r * υ
    B = 2π * dx
    source_contribution = -B * f(y) *∇G(xᵢ, y, dx)

    return boundary_contribution + source_contribution
end

function solvePoissonGradient(x::Point, Ω::Scene, f::Function, WoS_depth::Integer, num_Samples::Integer, ϵ::Real)
    num_Samples > 0 || throw(ArgumentError("num_Samples must be positive"))
    # Outputs are more accurate with high sample counts (≥500)
    return mean([walkOnSphere_PoissonGradient(x, Ω, f, WoS_depth, ϵ) for _ in 1:num_Samples])
end

