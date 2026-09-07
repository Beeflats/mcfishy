# 2D point
struct Point
    x::Float64
    y::Float64
end

# 2D vector
struct Vect
    x::Float64
    y::Float64
end

ZEROVECTOR = Vect(0, 0)

# Point translation
displace(P::Point, v::Vect) = Point(P.x + v.x, P.y + v.y)

⊕(P::Point, v::Vect) = displace(P, v)

# Direction vector from P to Q
→(P::Point, Q::Point) = Vect(Q.x - P.x, Q.y - P.y)

# Vect operations
Base.:+(v₁::Vect, v₂::Vect) = Vect(v₁.x + v₂.x, v₁.y + v₂.y)

Base.:-(v₁::Vect, v₂::Vect) = Vect(v₁.x - v₂.x, v₁.y - v₂.y)

Base.:-(v::Vect) = Vect(-v.x, -v.y)

Base.:*(c::Number, v::Vect) = Vect(c * v.x, c * v.y)

Base.:*(v::Vect, c::Number) = Vect(c * v.x, c * v.y)

Base.:/(v::Vect, c::Number) = Vect(v.x / c, v.y / c)

# Dot product
⋅(v₁::Vect, v₂::Vect) = v₁.x * v₂.x + v₁.y * v₂.y

# Length
length²(v::Vect) = v ⋅ v
norm(v::Vect) = sqrt(length²(v))
unit(v::Vect) = v / norm(v)

# Perpendicular vector
perp(v::Vect) = Vect(v.y, -v.x)

# Projection
proj(a::Vect, b::Vect) = (a ⋅ b) / length²(b) * b

oproj(a::Vect, b::Vect) = a - proj(a, b)

∥(v₁::Vect, v₂::Vect) = proj(v₁, v₂)

⟂(v₁::Vect, v₂::Vect) = oproj(v₁, v₂)

# Distance between 2D points
distance(P::Point, Q::Point) = norm(P → Q)

"""
Linear interpolation
"""
interpolate_linear(a::Point, b::Point, λ) = a ⊕ λ * (a → b)
interpolate_linear(a::Vect, b::Vect, λ) = a + λ * (b - a)
lerp = interpolate_linear
