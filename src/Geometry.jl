include("Vectors.jl")

∅ = nothing
∞ = Inf

"""
Geometry
"""
abstract type Geometry end

abstract type Geometry2D <: Geometry end
abstract type Geometry3D <: Geometry end

struct Ray
    #r(t) = O ⊕ t d
    origin::Point
    direction::Vec
end

ray(o::Point2, d::Vector2) = Ray(o, d)
ray(o::Point3, d::Vector3) = Ray(o, d)
(r::Ray)(t) = r.origin ⊕ t * r.direction

struct Line2D <: Geometry2D
    """
    A line characterized by a point p and direction vector d.
    The line is the set of points
        L(p,d) = { x ∈ ℝ² | x = p ⊕ t d }
    """
    point::Point2
    direction::Vector2
end

line(p::Point2, d::Vector2) = Line2D(p, d)
line(p₁::Point2, p₂::Point2) = Line2D(p₁, p₁ → p₂)
lineByNormal(p::Point2, n::Vector2) = Line2D(p, perp(n)) # L = { x ∈ ℝ² | (x → p) ⋅ n = 0 }

struct Circle <: Geometry2D
    """
    A circle is a line (not an area for our purposes) characterized by a center C
    and radius r.
        S¹(C, r) = { x ∈ ℝ² | (C → x) ⋅ (C → x) = r² }
    """
    center::Point2
    radius::Float64
end

circle(C::Point2, r::Float64) = Circle(C, r)
sphere(C::Point2, r::Float64) = Circle(C, r)

struct LineSegment2D <: Geometry2D
    """
    A line segment is the set of points between p₁ and p₂.
        L[p₁, p₂] = { q ∈ ℝ² | q = p₁ ⊕ λ(p₁→p₂)} where λ ∈ [0,1]
    It can be considered the the convex hull of two points in ℝ²
    """
    p₁::Point2
    p₂::Point2
end

lineSegment(p₁::Point2, p₂::Point2) = LineSegment2D(p₁, p₂)

struct Sphere <: Geometry3D
    """
    A sphere is a surface (not a volume for our purposes) characterized by a center C
    and radius r.
        S²(C, r) = { x ∈ ℝ³ | (C → x) ⋅ (C → x) = r² }
    """
    center::Point3
    radius::Float64
end

sphere(C::Point3, r::Float64) = Sphere(C, r)

struct Plane <: Geometry3D
    """
    A plane is a set of points characterized by a point p and normal vector n.
        Π(p, n) = { x ∈ ℝ³ | (x → p) ⋅ n = 0}
    """
    point::Point3
    normal::Vector3
end

struct LineSegment3D <: Geometry3D
    """
    A line segment is the set of points between p₁ and p₂.
        L[p₁, p₂] = { q ∈ ℝ³ | q = p₁ ⊕ λ(p₁→p₂)} where λ ∈ [0,1]
    It forms the edge of a mesh.
    """
    p₁::Point3
    p₂::Point3
end

lineSegment(p₁::Point3, p₂::Point3) = LineSegment3D(p₁, p₂)
edge(p₁::Point3, p₂::Point3) = LineSegment3D(p₁, p₂)

struct Triangle <: Geometry3D
    """
    The convex hull of three points in ℝⁿ
    T[p₁, p₂, p₃] = { q ∈ ℝⁿ | q = p₁ ⊕ (λ₁(p₁→p₂) + λ₂(p₁→p₃))} where λ₁,λ₁ ≥ 0 and λ₁ + λ₂ ∈ [0,1]
    """
    p₁::Point3
    p₂::Point3
    p₃::Point3
end

plane(T::Triangle) = Plane(T.p₁, (T.p₁ → T.p₂) × (T.p₁ → T.p₃))
area(T::Triangle) = norm((T.p₁ → T.p₂) × (T.p₁ → T.p₃)) / 2
function ∈(p::Point, T::Triangle)
    """
    A tetrahedron inequality states that for points P,Q,R and p that
        |PQR| ≤ |PQp| + |PpR| + |pQR|
    where |T| is denotes area of the triangle T.
    The point p is in the triangle PQR if and only if
        |PQR| = |PQp| + |PpR| + |pQR|
    """
    ε = 0.0001 # tolerance
    PQR = area(T)
    PQp = area(Triangle(T.p₁, T.p₂, p))
    PpR = area(Triangle(T.p₁, p, T.p₃))
    pQR = area(Triangle(p, T.p₂, T.p₃))
    # the computation for this function is inefficient since it involves calculating three cross products
    if abs(PQR - (PQp + PpR + pQR)) < ε
        return true
    else
        return false
    end
end

# TODO: 2D triangle, 3D lines, 3D line segments

# Nearest point queries
function nearestPoint(x::Point2, S::Circle)
    C = S.center
    R = S.radius
    if C == x                                             # if x is the centre point...
        return C ⊕ (r * unit(Vector2(randn(), randn()))) # perturb x randomly and pick the new nearest point 
    end
    return C ⊕ R * (C →ᵘ x)
end

function nearestPoint(x::Point2, L::Line2D)
    p = L.point
    d = L.direction
    return p ⊕ ((p → x) ∥ d)
end

function nearestPoint(x::Point2, L::LineSegment2D)
    P = L.p₁; Q = L.p₂
    if P == Q
        return P
    end
    PQ = P → Q
    Px = P → x
    λ = (Px ⋅ PQ) / norm²(PQ) # scalar multiple coefficient of Px ∥ PQ
    # clamp to segment
    t = clamp(λ, 0, 1)
    return P ⊕ t * PQ
end

function nearestPoint(x::Point3, L::LineSegment3D)
    P = L.p₁; Q = L.p₂
    if P == Q 
        return P
    end
    PQ = P → Q
    Px = P → x
    λ = (Px ⋅ PQ) / norm²(PQ) # scalar multiple coefficient of Px ∥ PQ
    t = clamp(λ, 0, 1) # clamp to segment
    return P ⊕ t * PQ
end

function nearestPoint(x::Point3, S::Sphere)
    C = S.center
    R = S.radius
    if C == x
        return C ⊕ (r * unit(Vector3(randn(), randn(), randn()))) # perturb x and pick the new nearest point
    end
    return C ⊕ R * (C →ᵘ x)
end

function nearestPoint(x::Point3, Π::Plane)
    p = Π.point
    n̂ = unit(Π.normal)
    return x ⊕ -((p → x) ∥ n̂)
end

function nearestPoint(x::Point3, T::Triangle)
    Π = plane(T)
    q = nearestPoint(x, Π)
    if q ∈ T
        return q
    else
        candidates = (nearestPoint(x, LineSegment3D(T.p₁, T.p₂)), # collect the closest points along each triangle edge
                      nearestPoint(x, LineSegment3D(T.p₂, T.p₃)),
                      nearestPoint(x, LineSegment3D(T.p₃, T.p₁)))
        return argmin(q -> distance(x, q), candidates)
    end
end

# Distance queries
distance(x::Point, g::Geometry) = distance(x, nearestPoint(x, g))