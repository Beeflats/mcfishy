"""
Example implementation of the WoS solvers
"""
include("./src/Solvers.jl")
include("./src/Render.jl")

# Scene
circle₁ = Circle(Point2(-2.5,  2.0), 1.2)
bc₁(x) = 2.0 + sin(4x.x) + 0.5cos(7x.y)

circle₂ = Circle(Point2( 2.0,  2.2), 1.0)
bc₂(x) = -1.0 + 1.5cos(5x.x) - 0.7sin(3x.y)

circle₃ = Circle(Point2(-2.0, -2.0), 1.4)
bc₃(x) = 0.5 + 2.0sin(6x.x + 2x.y) + 0.5cos(9x.y)

circle₄ = Circle(Point2( 2.5, -1.8), 0.9)
bc₄(x) = -2.0 + cos(8x.x) + 0.8sin(5x.y)

line₁ = LineSegment2D(Point2(-3.5, 2.0), Point2(3.0, 2.2))
bc_line(x) = 1.5sin(4x.x) + 0.8cos(3x.y) + 0.3x.x

∂𝕊 = scene(Dirichlet(circle₁, bc₁), 
           Dirichlet(circle₂, bc₂), 
           Dirichlet(circle₃, bc₃), 
           Dirichlet(circle₄, bc₄), 
           Dirichlet(line₁, bc_line))

# Walk-on-Spheres parameters
WoS_depth = 30     # Increase this for better chance of convergence
num_Samples = 200  # Increase this for more WoS samples
ϵ = 0.004          # Decrease this to decrease the thickness of ∂𝕊

# Rendering domain
Ω = makeDomain(10.0, 10.0)

resolution = (256, 256)
Nx, Ny = resolution
♯Ω = discretize(Ω, Nx, Ny)

# Scene preview
function viewBoundary(∂𝕊::Scene, domain::RenderDomain, resolution::Tuple, ϵ::Float64, colorscheme = defaultColorScheme)
    Nx, Ny = resolution
    ♯domain = discretize(domain, Nx, Ny)
    image = [begin x = ♯domain.grid[i, j]
                   candidates = [begin x_nearest = nearestPoint(x, bndry)
                                       bndry(x_nearest)
                                 end for bndry ∈ ∂𝕊.boundaries if distance(x, bndry) ≤ ϵ]
                   isempty(candidates) ? 0.0 : candidates[argmax(abs.(candidates))]
             end for i ∈ 1:Nx, j ∈ 1:Ny]

    m = maximum(abs, image)
    normalized = m == 0 ? zeros(size(image)) : (image ./ m .+ 1) ./ 2
    image = [get(colorscheme, normalized[i, j]) for i ∈ axes(normalized, 1), j ∈ axes(normalized, 2)]
    ImageView.imshow(image)
end

viewBoundary(∂𝕊, Ω, resolution, 0.02, ColorSchemes.twilight)

# Localised sources and sinks
A = Point2(1.5,  1.0); B = Point2(-1.8, -1.5); C = Point2(-0.5,  2.0) # Source and sink points
gaussian(μ::Point, σ::Real) = x -> exp(-norm²(μ → x) / (2σ^2))

f₁(x::Point) = 8.0 * gaussian(A, 0.7)(x) - 6.0 * gaussian(B, 0.5)(x) + 4.0 * gaussian(C, 0.3)(x)

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
#image_laplace = render(u_laplace, ♯Ω) # Render the scalar field with a discretized domain

"""
Solve the equation
    Δu = f,  where x ∈ Ω
with boundary condition
    u(x) = bc(x) with x ∈ ∂𝕊
"""
f = f₁
u_poisson(p::Point) = solvePoisson(p, ∂𝕊, f, WoS_depth, num_Samples, ϵ)
image_poisson = render(u_poisson, ♯Ω) # Render the scalar field with a discretized domain

u_poissonGradient(p::Point) = solvePoissonGradient(p, ∂𝕊, x -> 0.05 * f(x), WoS_depth, num_Samples, ϵ)
image_poissongrad = render(u_poissonGradient, ♯Ω) # Render the scalar field with a discretized domain
image_normpoissongrad = norm.(image_poissongrad);

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

image = process(image_normpoissongrad); # choose from image_laplace, image_poisson or image_normpoissongrad
viewImage(image, ColorSchemes.twilight)

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