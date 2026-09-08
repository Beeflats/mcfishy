include("Vectors.jl")

∅ = nothing
∞ = Inf

"""
Samples a random unit vector (circle) in 2D space.
"""
function uniformSampleCircle()
    return unit(Vect(randn(), randn()))
end

"""
Three shapes of concern for 2D meshes:
    - circles
    - lines
    - line segments
These provide nearest-point operations for 2D geometry.
"""
abstract type Geometry end

struct Ray
    """
    A ray in 2D space.
    r(t) = O ⊕ t d
    """
    origin::Point
    direction::Vect
end

(r::Ray)(t) = r.origin ⊕ t * r.direction

struct Line <: Geometry
    """
    A line characterized by a point p and direction vector d.
    The line is the set of points
        L(p,d) = { x ∈ ℝ² | x = p ⊕ t d }
    """
    point::Point
    direction::Vect
end

struct Circle <: Geometry
    """
    A circle is a surface characterized by a center C
    and radius r.
        C(C,r) = { x ∈ ℝ² | |C → x|² = r² }
    """
    center::Point
    radius::Float64
end

struct LineSegment <: Geometry
    """
    A line segment is the set of points between p₁ and p₂.
        S(p₁,p₂) = { p₁ ⊕ t(p₁ → p₂) | 0 ≤ t ≤ 1 }
    """
    p₁::Point
    p₂::Point
end


# Nearest point queries
function nearestPoint(x::Point, c::Circle)
    CX = c.center → x
    # x is exactly at the centre: there is no unique nearest point.
    if length²(CX) == 0
        return c.center ⊕ c.radius * î₂
    end
    return c.center ⊕ c.radius * unit(CX)
end

function nearestPoint(x::Point, l::Line
    d = l.point → x
    projected = d ∥ l.direction
    return l.point ⊕ projected
end

function nearestPoint(x::Point, s::LineSegment)
    d = s.p₁ → s.p₂
    v = s.p₁ → x
    t = (v ⋅ d) / length²(d)
    t = clamp(t, 0.0, 1.0)
    return s.p₁ ⊕ t * d
end

# Distance queries
distance(x::Point, g::Geometry) = distance(x, nearestPoint(x, g))
distance(x::Point, y::Point) = norm(x → y)

# Interior points
function inInterior(x::Point, c::Circle)
    return distance(c.center, x) ≤ c.radius
end
