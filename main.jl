"""
Example implementation of the WoS solvers
"""

include("LaplaceSolver.jl")
include("PoissonSolver.jl")
include("Render.jl")

# Scene
circle₁ = Circle(Point(-2.5,  2.0), 1.2)
bc₁(x) = 2.0 + sin(4x.x) + 0.5cos(7x.y)

circle₂ = Circle(Point( 2.0,  2.2), 1.0)
bc₂(x) = -1.0 + 1.5cos(5x.x) - 0.7sin(3x.y)

circle₃ = Circle(Point(-2.0, -2.0), 1.4)
bc₃(x) = 0.5 + 2.0sin(6x.x + 2x.y) + 0.5cos(9x.y)

circle₄ = Circle(Point( 2.5, -1.8), 0.9)
bc₄(x) = -2.0 + cos(8x.x) + 0.8sin(5x.y)

line = LineSegment(Point(-3.5, 2.0), Point(3.0, 2.2))
bc_line(x) = 1.5sin(4x.x) + 0.8cos(3x.y) + 0.3x.x

∂𝕊 = scene((circle₁, bc₁),
           (circle₂, bc₂),
           (circle₃, bc₃),
           (circle₄, bc₄),
           (line, bc_line))

# Walk-on-Spheres parameters
WoS_depth = 50     # Increase this for better chance of convergence
num_Samples = 500  # Increase this for more WoS samples
ϵ = 0.004          # Decrease this to decrease the thickness of ∂𝕊

# Rendering domain
Ω = makeDomain(10.0, 10.0)

resolution = (256, 256)
Nx, Ny = resolution
♯Ω = discretize(Ω, Nx, Ny)

# Localised sources and sinks
f₁(x::Point) = 8.0 * exp(-((x.x - 1.5)^2 + (x.y - 1.0)^2) / 0.7) -
               6.0 * exp(-((x.x + 1.8)^2 + (x.y + 1.5)^2) / 0.5) +
               4.0 * exp(-((x.x + 0.5)^2 + (x.y - 2.0)^2) / 0.3)

f₂(x::Point) = 3.0 * sin(2.5x.x) * cos(2.0x.y) + # This will blow up the solver
               2.0 * cos(4.0x.x + 3.0x.y) * exp(-(x.x^2 + x.y^2) / 18.0) 

# Solve
"""
Solve the equation
    Δu = 0,  where x ∈ Ω
with boundary condition
    u(x) = bc(x) with x ∈ ∂𝕊
"""
u_laplace(p::Point) = solveLaplace(p, ∂𝕊, WoS_depth, num_Samples, ϵ)
image_laplace = render(u_laplace, ♯Ω) # Render the scalar field with a discretized domain

"""
Solve the equation
    Δu = f,  where x ∈ Ω
with boundary condition
    u(x) = bc(x) with x ∈ ∂𝕊
"""
f = f₁
u_poisson(p::Point) = solvePoisson(p, ∂𝕊, f, WoS_depth, num_Samples, ϵ)
image_poisson = render(u_poisson, ♯Ω) # Render the scalar field with a discretized domain

# Visualise
image = image_laplace
viewImage(image, ColorSchemes.magma)

# Other implementations to try/PDEs to solve:
# Screened Poission: Δu - λu = f
# Linearised Poisson-Boltzmann: Δu - κ²u = -f
# Mixed and Neumann BVPs: ∂u/∂n = g
# Robin condition: αu + β ∂u/∂n = g
# Fractional laplacian: (-Δ)^(α/2)[u] = f, 0<α<2
# Variable diffusion: ∇⋅(D(x)∇u) = f or tr(A(x)Δu) + b(x)⋅∇u - c(x)u = f
# Diffusion-advection-reaction: DΔu + b⋅∇u - cu = f
# Biharmonic equation: Δ²u = 0
# Walk on Stars: star-shaped domains
# Interface/transmission PDEs
# Walk on Boundary
