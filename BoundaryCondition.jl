include("Geometry.jl")

# Boundary-condition objects
struct Object
    geometry::Geometry
    u::Function
end

object(geometry::Geometry, u::Function) = Object(geometry, u)

object(geometry::Geometry, c::Number) = Object(geometry, x -> c)

object(geometry::Geometry, v::Vect) = Object(geometry, x -> v)

(o::Object)(x::Point) = o.u(x) # Evaluate the boundary condition associated with an object.

# Convenience constructors
Dirichlet(geometry::Geometry, c::Number) = object(geometry, c)
Dirichlet(geometry::Geometry, v::Vect) = object(geometry, v)
Dirichlet(geometry::Geometry, u::Function) = object(geometry, u)

# Scene
struct Scene
    objects::Vector{Object}
end

scene() = Scene(Object[])
scene(o::Object...) = Scene(Object[o...])
scene(objects::Tuple...) = Scene(Object[object(geometry, u) for (geometry, u) ∈ objects])

scene(mesh::AbstractVector{<:Geometry}, u::Function) = Scene(Object[object(geometry, u) for geometry ∈ mesh])
scene(mesh::AbstractVector{<:Geometry}, c::Number) = scene(mesh, x -> c)
scene(mesh::AbstractVector{<:Geometry}, v::Vect) = scene(mesh, x -> v)

# Adding objects to a scene
Base.:∪(𝕊::Scene, o::Object) = Scene([𝕊.objects... o])
Base.:∪(𝕊::Scene, objects::Vector{Object}) = Scene([𝕊.objects... objects...])
Base.:∪(𝕊₁::Scene, 𝕊₂::Scene) = Scene([𝕊₁.objects... 𝕊₂.objects...])
Base.:∪(o::Object, 𝕊::Scene) = 𝕊 ∪ o
Base.:∪(o₁::Object, o₂::Object) = scene(o₁, o₂)

# Distance to objects
distance(x::Point, obj::Object) = distance(x, obj.geometry)
nearestPoint(x::Point, obj::Object) = nearestPoint(x, obj.geometry)

# Distance to scene
function distance(x::Point, 𝕊::Scene)
    isempty(𝕊.objects) && throw(ArgumentError("Cannot compute distance to an empty Scene"))
    return minimum(distance(x, obj) for obj ∈ 𝕊.objects)
end

# Nearest object and point
function nearest(x::Point, 𝕊::Scene)
    isempty(𝕊.objects) && throw(ArgumentError("Cannot find nearest object in an empty Scene"))
    distances = [distance(x, obj) for obj ∈ 𝕊.objects]
    obj_nearest = 𝕊.objects[argmin(distances)]
    x_nearest = nearestPoint(x, obj_nearest)
    return x_nearest, obj_nearest
end
