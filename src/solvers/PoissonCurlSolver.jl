include("PoissonGradientSolver.jl")

"""
Solve the equation
    Δu = f,  where x ∈ Ω
with boundary condition
    u(x) = bc(x) with x ∈ ∂Ω

NOTE: curl(u) can only be computed if u is a vector field
"""

function solvePoissonCurl(x::Point2, Ω::Scene, f::Function, WoS_depth::Integer, num_Samples::Integer, ϵ::Float64)
   ∇u = solvePoissonGradient(x, Ω, f, WoS_depth, num_Samples, ϵ)
    return ∇u.i.y - ∇u.j.x
end

function convertGradToCurl(∇F::Endomorphism3)
    return Vector3(∇F.j.z - ∇F.k.y,
                   ∇F.k.x - ∇F.i.z,
                   ∇F.i.y - ∇F.j.x)
end

function solvePoissonCurl(x::Point3, Ω::Scene, f::Function, WoS_depth::Integer, num_Samples::Integer, ϵ::Float64)
    ∇u = solvePoissonGradient(x, Ω, f, WoS_depth, num_Samples, ϵ)
    return convertGradToCurl(∇u)
end