include("Geometry.jl")

# Boundary-condition objects
abstract type Boundary end

struct Boundary2D <: Boundary
    geometry::Geometry2D
    u::Function
end

struct Boundary3D <: Boundary
    geometry::Geometry3D
    u::Function
end

boundary(geometry::Geometry2D, u::Function) = Boundary2D(geometry, u)
boundary(geometry::Geometry3D, u::Function) = Boundary3D(geometry, u)
boundary(geometry::Geometry2D, c::Number) = Boundary2D(geometry, x -> c)
boundary(geometry::Geometry3D, c::Number) = Boundary3D(geometry, x -> c)
boundary(geometry::Geometry2D, v::Vec) = Boundary2D(geometry, x -> v)
boundary(geometry::Geometry3D, v::Vec) = Boundary3D(geometry, x -> v)

(o::Boundary)(x::Point) = o.u(x) # Evaluate the boundary condition.

# Convenience constructors
Dirichlet(geometry::Geometry, c::Number) = boundary(geometry, c)
Dirichlet(geometry::Geometry, v::Vec) = boundary(geometry, v)
Dirichlet(geometry::Geometry, u::Function) = boundary(geometry, u)

# Scene
abstract type Scene end

struct Scene2D <: Scene
    boundaries::Vector{Boundary2D}
end

struct Scene3D <: Scene
    boundaries::Vector{Boundary3D}
end

scene2D() = Scene2D(Boundary2D[])
scene3D() = Scene3D(Boundary3D[])
scene(o::Boundary2D...) = Scene2D(Boundary2D[o...])
scene(o::Boundary3D...) = Scene3D(Boundary3D[o...])
scene(boundaries::Tuple...) = scene((boundary(geometry, u) for (geometry, u) in boundaries)...)

scene(mesh::AbstractVector{<:Geometry}, u::Function) = scene((boundary(geometry, u) for geometry ∈ mesh)...)
scene(mesh::AbstractVector{<:Geometry}, c::Number) = scene(mesh, x -> c)
scene(mesh::AbstractVector{<:Geometry}, v::Vec) = scene(mesh, x -> v)

# Adding boundaries to a scene
Base.:∪(𝕊::Scene2D, o::Boundary2D) = Scene2D([𝕊.boundaries..., o])
Base.:∪(𝕊::Scene3D, o::Boundary3D) = Scene3D([𝕊.boundaries..., o])
Base.:∪(𝕊::Scene, boundaries::Vector{Boundary}) = Scene([𝕊.boundaries... boundaries...])
Base.:∪(𝕊₁::Scene2D, 𝕊₂::Scene2D) = Scene2D([𝕊₁.boundaries..., 𝕊₂.boundaries...])
Base.:∪(𝕊₁::Scene3D, 𝕊₂::Scene3D) = Scene3D([𝕊₁.boundaries..., 𝕊₂.boundaries...])
Base.:∪(o::Boundary, 𝕊::Scene) = 𝕊 ∪ o
Base.:∪(o₁::Boundary, o₂::Boundary) = scene(o₁, o₂)
Base.:∪(::Scene2D, ::Scene3D) = throw(ArgumentError("Cannot combine 2D and 3D scenes"))
Base.:∪(::Scene3D, ::Scene2D) = throw(ArgumentError("Cannot combine 2D and 3D scenes"))

# Distance to boundaries
distance(x::Point, bndry::Boundary) = distance(x, bndry.geometry)
nearestPoint(x::Point, bndry::Boundary) = nearestPoint(x, bndry.geometry)

# Distance to scene
function distance(x::Point, 𝕊::Scene)
    isempty(𝕊.boundaries) && throw(ArgumentError("Cannot compute distance to an empty Scene"))
    return minimum(distance(x, bndry) for bndry ∈ 𝕊.boundaries)
end

# Nearest boundary and point
function nearest(x::Point, 𝕊::Scene)
    isempty(𝕊.boundaries) && throw(ArgumentError("Cannot find nearest boundary in an empty Scene"))
    distances = [distance(x, bndry) for bndry ∈ 𝕊.boundaries]
    bndry_nearest = 𝕊.boundaries[argmin(distances)]
    x_nearest = nearestPoint(x, bndry_nearest)
    return x_nearest, bndry_nearest
end

