"""
Points
"""
abstract type Point end # A Point is an element of Euclidean metric space 𝔼ⁿ

struct Point2 <: Point
    x::Float64
    y::Float64
end

struct Point3 <: Point
    x::Float64
    y::Float64
    z::Float64
end

point(x, y) = Point2(x, y)
point(x, y, z) = Point3(x, y, z)
dim(::Point2) = 2
dim(::Point3) = 3 

# Euclidean metric axiom
distance²(P::Point2, Q::Point2) = (P.x-Q.x)^2 + (P.y-Q.y)^2
distance²(P::Point3, Q::Point3) = (P.x-Q.x)^2 + (P.y-Q.y)^2 + (P.z-Q.z)^2
distance(P::Point, Q::Point) = √distance²(P, Q)

"""
Vectors
"""
abstract type Vec end # A Vector (Vec) is an element of a vector space ℝⁿ

struct Vector2 <: Vec
    x::Float64
    y::Float64
end

struct Vector3 <: Vec
    x::Float64
    y::Float64
    z::Float64
end

vector(x, y) = Vector2(x, y)
vector(x, y, z) = Vector3(x, y, z)
dim(::Vector2) = 2
dim(::Vector3) = 3 

# Vector space axioms
# Additive identity
const ZEROVECTOR₂ = Vector2(0, 0)
const ZEROVECTOR₃ = Vector3(0, 0, 0)
# Closure, commutativity and associativity of addition
Base.:+(v₁::Vector2, v₂::Vector2) = Vector2(v₁.x + v₂.x, v₁.y + v₂.y)
Base.:+(v₁::Vector3, v₂::Vector3) = Vector3(v₁.x + v₂.x, v₁.y + v₂.y, v₁.z + v₂.z)
Base.:-(v₁::Vector2, v₂::Vector2) = Vector2(v₁.x - v₂.x, v₁.y - v₂.y)
Base.:-(v₁::Vector3, v₂::Vector3) = Vector2(v₁.x - v₂.x, v₁.y - v₂.y, v₁.z - v₂.z)
# Negation
Base.:-(v::Vector2) = Vector2(-v.x, -v.y)
Base.:-(v::Vector3) = Vector3(-v.x, -v.y, -v.z)
# Closure, distributivity and associativity of scalar multiplication
Base.:*(c::Number, v::Vector2) = Vector2(c * v.x, c * v.y)
Base.:*(c::Number, v::Vector3) = Vector3(c * v.x, c * v.y, c * v.z)
Base.:*(v::Vector2, c::Number) = Vector2(c * v.x, c * v.y)
Base.:*(v::Vector3, c::Number) = Vector3(c * v.x, c * v.y, c * v.z)
Base.:/(v::Vector2, c::Number) = Vector2(v.x / c, v.y / c)
Base.:/(v::Vector3, c::Number) = Vector3(v.x / c, v.y / c, v.z / c)

# Displacement axioms:
 # Displacement space is the space (𝔼ⁿ, ℝⁿ, ⊕, →) in which
 # ⊕ [\oplus] and → [\to] are binary operators which satisfying the following
 #   P ⊕ v ∈ 𝔼ⁿ
 #   P → Q ∈ ℝⁿ
 #   P ⊕ (P → Q) = Q
displace(P::Point2, v::Vector2) = Point2(P.x + v.x, P.y + v.y)
displace(P::Point3, v::Vector2) = Point3(P.x + v.x, P.y + v.y, P.z + v.z)
⊕(P::Point, v::Vec) = displace(P, v)
# The third axiom can be reformulated as P → (P ⊕ v) = v, therefore
→(P::Point2, Q::Point2) = Vector2(Q.x - P.x, Q.y - P.y)
→(P::Point3, Q::Point3) = Vector2(Q.x - P.x, Q.y - P.y, Q.z - P.z)

# Dot product space axioms
⋅(v₁::Vector2, v₂::Vector2) = v₁.x * v₂.x + v₁.y * v₂.y
⋅(v₁::Vector3, v₂::Vector3) = v₁.x * v₂.x + v₁.y * v₂.y + + v₁.z * v₂.z

# Normed vector space axiom:
#   |P → Q| = d(P, Q)
norm²(v::Vec) = v ⋅ v
length²(v::Vec) = norm²(v) # norm(v) was defined to be equivalent to distance(O, O ⊕ v) where O is an arbitrary point in 𝔼ⁿ 
norm(v::Vec) = sqrt(norm²(v))
unit(v::Vec) = v / norm(v)
normalize(v::Vec) = unit(v)
→ᵘ(P::Point, Q::Point) = unit(P → Q)

# Cross product
function ×(v₁::Vector3, v₂::Vector3)
	return Vector3(v₁.y * v₂.z - v₁.z * v₂.y,
		           v₁.z * v₂.x - v₁.x * v₂.z,
		           v₁.x * v₂.y - v₁.y * v₂.x)
end

# Orthonormal basis Vectors
const î₂ = Vector2(1, 0)
const ĵ₂ = Vector2(0, 1)

const î = Vector3(1, 0, 0)
const ĵ = Vector3(0, 1, 0)
const k̂ = Vector3(0, 0, 1)

# Arbitrary perpendicular vector
perp(v::Vector2) = Vector2(v.y, -v.x)

# Projection
proj(a::Vec, b::Vec) = (a ⋅ b) / length²(b) * b
oproj(a::Vec, b::Vec) = a - proj(a, b)
∥(v₁::Vec, v₂::Vec) = proj(v₁, v₂)
⟂(v₁::Vec, v₂::Vec) = oproj(v₁, v₂)

"""
Linear combination of vectors 
"""

using LinearAlgebra: rank
function solveLinearCombination(v₁::Vector2, v₂::Vector2, u::Vector2)
	"""
	Find the coefficients λ₁,λ₂ that satisfy the equation
		λ₁v₁ + λ₂v₂ = u
    Applications: 
        - Determines the vector components (λ₁,λ₂) of u in terms of the basis vectors v₁,v₂
	"""
    # Convert the problem into computing the reduced row echelon form
    # The linear combination is reformulated as Vλ = u, and solve for λ
    V = [v₁.x v₂.x;
         v₁.y v₂.y]
    if rank(V) < 2
        return nothing
    end
    return V \ [u.x; u.y]
end

function solveLinearCombination(v₁::Vector3, v₂::Vector3, v₃::Vector3, u::Vector3)
	"""
	Find the coefficients λ₁,λ₂,λ₃ that satisfy the equation
		λ₁v₁ + λ₂v₂ + λ₃v₃ = u
    Applications: 
        - Determines the vector components (λ₁,λ₂,λ₃) of u in terms of the basis vectors v₁,v₂,v₃
	"""
    # Convert the problem into computing the reduced row echelon form
    # The linear combination is reformulated as Vλ = u, and solve for λ
    V = [v₁.x v₂.x v₃.x;
         v₁.y v₂.y v₃.y;
         v₁.z v₂.z v₃.z]
    if rank(V) < 3
        return nothing
    end
    return V \ [u.x; u.y; u.z]
end

"""
Endomorphisms
"""
abstract type Endomorphism end

struct Endomorphism2 <: Endomorphism
    """
    An endomorphism M: ℝ² → ℝ² is a linear transformation defined by
        M(î) = M.i and 
        M(ĵ) = M.j
    """
    i::Vector2
    j::Vector2
end

struct Endomorphism3 <: Endomorphism
    """
    An endomorphism M: ℝ³ → ℝ³ is a linear transformation defined by
        M(î) = M.i, 
        M(ĵ) = M.j, and 
        M(k̂) = M.k
    """
    i::Vector3
    j::Vector3
    k::Vector3
end

endomorphism(Mi::Vector2, Mj::Vector2) = Endomorphism2(Mi, Mj)
endomorphism(Mi::Vector3, Mj::Vector3, Mk::Vector3) = Endomorphism3(Mi, Mj, Mk)
endo(Mi::Vector2, Mj::Vector2) = endomorphism(Mi, Mj)
endo(Mi::Vector3, Mj::Vector3, Mk::Vector3) = endomorphism(Mi, Mj, Mk)

function endomorphism(v₁::Vector2, Mv₁::Vector2, 
                      v₂::Vector2, Mv₂::Vector2)
    """
    An endomorphism M in which
        M(v₁) = Mv₁ and M(v₂) = Mv₂.
    One may consider v₁,v₂ as basis vectors for ℝ². 

    # Derivation of the formula for the result:
        Mvᵢ = M(vᵢ)
            = M(vᵢ.x * î + vᵢ.y * ĵ
            = vᵢ.x * M(î) + vᵢ.y * M(ĵ)
        Thus we have two equations: 
         - Mv₁ = v₁.x * Mî + v₁.y * Mĵ
         - Mv₂ = v₂.x * Mî + v₂.y * Mĵ
        Then solve the set of linear equations for Mî, Mĵ. ∎
    """
    Δ = v₁.x * v₂.y - v₁.y * v₂.x
    iszero(Δ) && throw(ArgumentError("Inputs are linearly dependent"))
    Mî = ( v₂.y * Mv₁ - v₁.y * Mv₂) / Δ
    Mĵ = (-v₂.x * Mv₁ + v₁.x * Mv₂) / Δ
    return Endomorphism2(Mî, Mĵ)
end

function endomorphism(v₁::Vector3, Mv₁::Vector3,
                      v₂::Vector3, Mv₂::Vector3,
                      v₃::Vector3, Mv₃::Vector3)
    """
    An endomorphism M in which
        M(v₁) = Mv₁,
        M(v₂) = Mv₂, and
        M(v₃) = Mv₃.
    One may consider v₁,v₂,v₃ as basis vectors for ℝ³.

    # Derivation of the formula for the result:
        Mvᵢ = M(vᵢ)
            = M(vᵢ.x * î + vᵢ.y * ĵ + vᵢ.z * k̂)
            = vᵢ.x * M(î) + vᵢ.y * M(ĵ) + vᵢ.z * M(k̂)
        Thus we have three equations: 
         - Mv₁ = v₁.x * Mî + v₁.y * Mĵ + v₁.z * Mk̂
         - Mv₂ = v₂.x * Mî + v₂.y * Mĵ + v₂.z * Mk̂
         - Mv₃ = v₃.x * Mî + v₃.y * Mĵ + v₃.z * Mk̂
        Then solve the set of linear equations for Mî, Mĵ, Mk̂. ∎
    """
    Δ = v₁.x * (v₂.y * v₃.z - v₂.z * v₃.y) -
        v₁.y * (v₂.x * v₃.z - v₂.z * v₃.x) +
        v₁.z * (v₂.x * v₃.y - v₂.y * v₃.x)
    iszero(Δ) && throw(ArgumentError("Inputs are linearly dependent"))
    Mî = ((v₂.y * v₃.z - v₂.z * v₃.y) * Mv₁ +
          (v₃.y * v₁.z - v₃.z * v₁.y) * Mv₂ +
          (v₁.y * v₂.z - v₁.z * v₂.y) * Mv₃) / Δ
    Mĵ = ((v₂.z * v₃.x - v₂.x * v₃.z) * Mv₁ +
          (v₃.z * v₁.x - v₃.x * v₁.z) * Mv₂ +
          (v₁.z * v₂.x - v₁.x * v₂.z) * Mv₃) / Δ
    Mk̂ = ((v₂.x * v₃.y - v₂.y * v₃.x) * Mv₁ +
          (v₃.x * v₁.y - v₃.y * v₁.x) * Mv₂ +
          (v₁.x * v₂.y - v₁.y * v₂.x) * Mv₃) / Δ
    return Endomorphism3(Mî, Mĵ, Mk̂)
end

function endomorphism(λ₁::Number, v₁::Vector2,
                      λ₂::Number, v₂::Vector2)
    """
    An endomorphism M in which
        M(v₁) = λ₁v₁, and
        M(v₂) = λ₂v₂
    The vectors v₁,v₂ the eigenvectors of the endomorphism with eigenvalues λ₁,λ₂ respectively.
    """
    # Complex eigenvalues are not permitted
    return endomorphism(v₁, λ₁*v₁, v₂, λ₂*v₂)
end

function endomorphism(λ₁::Number, v₁::Vector3,
                      λ₂::Number, v₂::Vector3,
                      λ₃::Number, v₃::Vector3)
    """
    An endomorphism M in which
        M(v₁) = λ₁v₁,
        M(v₂) = λ₂v₂, and 
        M(v₃) = λ₃v₃
    The vectors v₁,v₂,v₃ the eigenvectors of the endomorphism with eigenvalues λ₁,λ₂,λ₃ respectively.
    """
    # Complex eigenvalues are not permitted
    return endomorphism(v₁, λ₁*v₁, v₂, λ₂*v₂, v₃, λ₃*v₃)
end

# Vector space axioms for endomorphisms
# Additive identity
const ZEROENDOMORPHISM₂ = Endomorphism2(ZEROVECTOR₂, ZEROVECTOR₂)
const ZEROENDOMORPHISM₃ = Endomorphism3(ZEROVECTOR₃, ZEROVECTOR₃, ZEROVECTOR₃) 
# Addition
Base.:+(M₁::Endomorphism2, M₂::Endomorphism2) = Endomorphism2(M₁.i + M₂.i, M₁.j + M₂.j)
Base.:+(M₁::Endomorphism3, M₂::Endomorphism3) = Endomorphism3(M₁.i + M₂.i, M₁.j + M₂.j, M₁.k + M₂.k)
Base.:-(M₁::Endomorphism2, M₂::Endomorphism2) = Endomorphism2(M₁.i - M₂.i, M₁.j - M₂.j)
Base.:-(M₁::Endomorphism3, M₂::Endomorphism3) = Endomorphism3(M₁.i - M₂.i, M₁.j - M₂.j, M₁.k - M₂.k)
Base.:-(M::Endomorphism2) = Endomorphism2(-M.i, -M.j)
Base.:-(M::Endomorphism3) = Endomorphism3(-M.i, -M.j, -M.k)
# Homogeneity of scalar multiplication
Base.:*(c::Number, M::Endomorphism2) = Endomorphism2(c*M.i, c*M.j)
Base.:*(c::Number, M::Endomorphism3) = Endomorphism3(c*M.i, c*M.j, c*M.k)
Base.:*(M::Endomorphism2, c::Number) = Endomorphism2(c*M.i, c*M.j)
Base.:*(M::Endomorphism3, c::Number) = Endomorphism3(c*M.i, c*M.j, c*M.k)
Base.:/(M::Endomorphism2, c::Number) = Endomorphism2(M.i/c, M.j/c)
Base.:/(M::Endomorphism3, c::Number) = Endomorphism3(M.i/c, M.j/c, M.k/c)

"""
Linear transformations
   Endomorphisms have the property of linearity and homogeneity
     M(u + v) = M(u) + M(v)
 # Derivation of the vector transformation formula:
        M(v) = M(∑ᵢ[v.i * eᵢ])  # by definition of Vector struct
             = ∑ᵢ[v.i * M(eᵢ)]  # by linearity
             = ∑ᵢ[v.i * M.eᵢ]   # by definition of Endomorphism struct
    where vᵢ is the iᵗʰ component of the vector v under the basis set {e₁, e₂, ..., eₙ}
"""
# Vector transformation
Base.:*(M::Endomorphism2, v::Vector2) = v.x * M.i + v.y * M.j
Base.:*(M::Endomorphism3, v::Vector3) = v.x * M.i + v.y * M.j + v.z * M.k
    # NOTE: Not a multiplication; moreso evaluating a function
(M::Endomorphism)(v::Vec) = M * v
# Composition
∘(S::Endomorphism2, T::Endomorphism2) = Endomorphism2(S * T.i, S * T.j)
∘(S::Endomorphism3, T::Endomorphism3) = Endomorphism3(S * T.i, S * T.j, S * T.k )
Base.:*(S::Endomorphism, T::Endomorphism) = S ∘ T
# Identity transformation
ı₂ = Endomorphism2(î₂, ĵ₂) # [\imath\_2]
ı₃ = Endomorphism3(î, ĵ, k̂) # [\imath\_3]
function ı(d)
    if d == 2
        return I₂
    elseif d == 3
        return I₃
    end
end

# Inverse transformation
inv(M::Endomorphism2) = endomorphism(M.i, î₂, # M⁻¹(M(î)) = î
                                     M.j, ĵ₂) # M⁻¹(M(ĵ)) = ĵ
inv(M::Endomorphism3) = endomorphism(M.i, î, # M⁻¹(M(î)) = î
                                     M.j, ĵ, # M⁻¹(M(ĵ)) = ĵ
                                     M.k, k̂) # M⁻¹(M(k̂)) = k̂
# Left inverses are equivalent to right inverses so M(M⁻¹(v)) = v = ıv

# Outer product transformation
"""
An endomorphism M parameterized by two vectors
    M(w) = (u ⋅ w) * v
"""
outer(v::Vector2, u::Vector2) = Endomorphism2(u.x * v, u.y * v)
outer(v::Vector3, u::Vector3) = Endomorphism3(u.x * v, u.y * v, u.z * v)
⊗(v::Vec, u::Vec) = outer(v, u)
outer(v::Vec) = outer(v, v)
⊗(x::Number, y::Number) = x * y
⊗(x::Number, v::Vec) = x * v

# Endomorphism invariants
trace(M::Endomorphism2) = M.i.x + M.j.y
trace(M::Endomorphism3) = M.i.x + M.j.y + M.k.z
tr(M::Endomorphism) = trace(M)
det(M::Endomorphism2) = M.i.x * M.j.y - M.i.y * M.j.x
function det(M::Endomorphism3)
    return M.i.x * (M.j.y * M.k.z - M.j.z * M.k.y) -
           M.j.x * (M.i.y * M.k.z - M.i.z * M.k.y) +
           M.k.x * (M.i.y * M.j.z - M.i.z * M.j.y)
end
frobeniusNorm(M::Endomorphism2) = √(norm²(M.i) + norm²(M.j))
frobeniusNorm(M::Endomorphism3) = √(norm²(M.i) + norm²(M.j) + norm²(M.k))
norm(M::Endomorphism) = frobeniusNorm(M)
function spectralNorm(M::Endomorphism2)
    a = M.i ⋅ M.i
    b = M.i ⋅ M.j
    c = M.j ⋅ M.j
    return √((a + c + √((a - c)^2 + 4b^2)) / 2)
end
function spectralNorm(M::Endomorphism3)
    nothing
    # TODO: Spectral norm
end

"""
Linear interpolation
"""
interpolate_linear(a::Point, b::Point, λ) = a ⊕ λ * (a → b)
interpolate_linear(a::Vec, b::Vec, λ) = a + λ * (b - a)
lerp = interpolate_linear
