include("BoundaryCondition.jl")

mean(x) = sum(x) / length(x)

function uniformSampleDisk(center::Point, radius::Real)
    r = radius * sqrt(rand())
    ω = uniformSampleCircle()
    return center ⊕ r * ω
end

G(x, y, dx) = 1/(2π) * log(dx / distance(x,y))

"""
Solve the equation
    Δu = f,  where x ∈ Ω
with boundary condition
    u(x) = bc(x) with x ∈ ∂Ω
"""

function walkOnSphere_Poisson(xᵢ::Point, ∂Ω::Scene, f::Function, WoS_depth::Integer,ϵ ::Real)
    WoS_depth ≥ 0 || throw(ArgumentError("WoS_depth must be non-negative"))
    dx = distance(xᵢ, ∂Ω)
    if WoS_depth == 0 || dx ≤ ϵ
        x_nearest, obj = nearest(xᵢ, ∂Ω)
        return obj(x_nearest)
    else
        ω = uniformSampleCircle()
        xᵢ₊₁ = xᵢ ⊕ dx * ω
        boundary_contribution = walkOnSphere_Poisson(xᵢ₊₁, ∂Ω, f, WoS_depth - 1, ϵ)

        υ = uniformSampleCircle()
        r = dx * sqrt(rand())
        y = xᵢ ⊕ r * υ
        B = 2π * dx
        source_contribution = -B * f(y) * G(xᵢ, y, dx)
    
        return boundary_contribution + source_contribution
    end
end

function solvePoisson(x::Point, Ω::Scene, f::Function, WoS_depth::Integer, num_Samples::Integer, ϵ::Real)
    num_Samples > 0 || throw(ArgumentError("num_Samples must be positive"))
    return mean([walkOnSphere_Poisson(x, Ω, f, WoS_depth, ϵ) for _ in 1:num_Samples])
end

