
include("./../Vectors.jl")
include("./../BoundaryCondition.jl")

# Mean
mean(x) = sum(x) / length(x)

# Sampling
uniformDirectionSample(::Val{2}) = unit(Vector2(randn(), randn())) # Unif(S¹)
uniformDirectionSample(::Val{3}) = unit(Vector3(randn(), randn(), randn())) # Unif(S²)
uniformDirectionSample(d::Integer) = uniformDirectionSample(Val(d)) # Unif(∂B)

function uniformBallPointSample(x::Point2, R::Float64)
    ρ = R * sqrt(rand())              # ρ = R√U where U ~ Unif([0,1])
    d = uniformDirectionSample(Val(2))      # d ~ Unif(S¹)
    return x ⊕ ρ * d                 # y ~ Unif(B(x, R))
end

function uniformBallPointSample(x::Point3, R::Float64)
    ρ = R * cbrt(rand())              # ρ = R∛U where U ~ Unif([0,1])
    d = uniformDirectionSample(Val(3))      # d ~ Unif(S²)
    return x ⊕ ρ * d                  # y ~ Unif(B(x, R))
end

# Perimeter and surface area
μ∂B(::Point2, r::Float64) = 2π * r
μ∂B(::Point3, r::Float64) = 4π * r^2

# Convolution kernels
# Harmonic Green's function of the Laplace operator on a ball or radius R
G(x::Point2, y::Point2, R::Float64) = 1/(2π) * log(R/distance(x,y))
∇G(x::Point2, y::Point2, R::Float64) = (x → y)/(2π) * (1/distance(x,y)^2 - 1/R^2)
#∫G(x::Point2, y::Point2, R::Float64) = R^2 / 4 
G(x::Point3, y::Point3, R::Float64) = 1/(4π) * (R - distance(x,y))/(R * distance(x,y))
∇G(x::Point3, y::Point3, R::Float64) = (x → y)/(4π) * (1/distance(x,y)^3 - 1/R^3)
#∫G(x::Point3, y::Point3, R::Float64) = R^2 / 6 

