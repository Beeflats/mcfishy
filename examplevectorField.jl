"""
Example implementation of the vector-valued WoS solvers
"""
include("./src/LaplaceSolver.jl")
include("./src/PoissonSolver.jl")
include("./src/PoissonGradientSolver.jl")
include("./src/PoissonCurlSolver.jl")
include("./src/Render.jl")

# Scene
circle₁ = Circle(Point2(-2.5,  2.0), 1.2)
bc₁(x::Point2) = Vector2(2.0 + sin(4x.x) + 0.5cos(7x.y),
                         1.0 + 0.8cos(3x.x) - 0.4sin(5x.y))

circle₂ = Circle(Point2( 2.0,  2.2), 1.0)
bc₂(x::Point2) = Vector2(-1.0 + 1.5cos(5x.x) - 0.7sin(3x.y),
                         0.5 + sin(4x.x + x.y))

circle₃ = Circle(Point2(-2.0, -2.0), 1.4)
bc₃(x::Point2) = Vector2(0.5 + 2.0sin(6x.x + 2x.y) + 0.5cos(9x.y),
                        -0.5 + cos(5x.x) + 0.7sin(4x.y))

circle₄ = Circle(Point2( 2.5, -1.8), 0.9)
bc₄(x::Point2) = Vector2(-2.0 + cos(8x.x) + 0.8sin(5x.y),
                          1.0 + 0.6sin(6x.x) - 0.5cos(3x.y))

line₁ = LineSegment2D(Point2(-3.5, 2.0), Point2(3.0, 2.2))
bc_line(x::Point2) = Vector2(1.5sin(4x.x) + 0.8cos(3x.y) + 0.3x.x,
                             0.7cos(3x.x) - 0.5sin(4x.y) + 0.2x.y)

∂𝕊 = scene(Dirichlet(circle₁, bc₁), 
           Dirichlet(circle₂, bc₂), 
           Dirichlet(circle₃, bc₃), 
           Dirichlet(circle₄, bc₄), 
           Dirichlet(line₁, bc_line))

# Walk-on-Spheres parameters
WoS_depth = 30
num_Samples = 300
ϵ = 0.004

# Rendering domain
Ω = makeDomain(10.0, 10.0)

resolution = (256, 256)
Nx, Ny = resolution
♯Ω = discretize(Ω, Nx, Ny)

# Vector-valued sources and sinks
f(x::Point2) = Vector2(
    8.0 * exp(-((x.x - 1.5)^2 + (x.y - 1.0)^2) / 0.7) -
    6.0 * exp(-((x.x + 1.8)^2 + (x.y + 1.5)^2) / 0.5) +
    4.0 * exp(-((x.x + 0.5)^2 + (x.y - 2.0)^2) / 0.3),

    -5.0 * exp(-((x.x + 1.0)^2 + (x.y - 1.5)^2) / 0.6) +
     3.0 * exp(-((x.x - 2.0)^2 + (x.y + 1.0)^2) / 0.4))

# Solve

"""
Solve the vector-valued equation
    Δu = 0,  where x ∈ Ω
with vector-valued boundary condition
    u(x) = bc(x),  where x ∈ ∂𝕊.
"""
u_laplace(p::Point2) = solveLaplace(p, ∂𝕊, WoS_depth, num_Samples, ϵ)
image_laplace = render(u_laplace, ♯Ω)
image_normlaplace = norm.(image_laplace)

"""
Solve the vector-valued equation
    Δu = f,  where x ∈ Ω
with vector-valued boundary condition
    u(x) = bc(x),  where x ∈ ∂𝕊.
"""
u_poisson(p::Point2) = solvePoisson(p, ∂𝕊, f, WoS_depth, num_Samples, ϵ)
image_poisson = render(u_poisson, ♯Ω)
image_normpoisson = norm.(image_poisson)

u_poissonGradient(p::Point2) = solvePoissonGradient(p, ∂𝕊, x -> 0.1 * f(x), WoS_depth, num_Samples, ϵ)
image_poissongrad = render(u_poissonGradient, ♯Ω)
image_detpoissongrad = det.(image_poissongrad)
image_fnormpoissongrad = frobeniusNorm.(image_poissongrad)
image_snormpoissongrad = spectralNorm.(image_poissongrad)

u_poissonCurl(p::Point2) = solvePoissonCurl(p, ∂𝕊, x -> 0.1 * f(x), WoS_depth, num_Samples, ϵ)
image_poissoncurl = render(u_poissonCurl, ♯Ω)

# Visualise
function process(image)
    # A quick, dirty method to remove NaNs from an image
    image = copy(image)
    for i ∈ axes(image, 1), j ∈ axes(image, 2)
        if isnan(image[i, j])
            image[i, j] = 0
        end
    end
    return image
end
image = process(image_poissoncurl) # choose from image_normlaplace, image_normpoisson, image_detpoissongrad, image_fnormpoissongrad, image_snormpoissongrad and image_poissoncurl
viewImage(image, ColorSchemes.twilight)