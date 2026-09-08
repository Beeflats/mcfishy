# GPU Implementation - Made by feeding the CPU code to ChatGPT and pasting the translated output
# Restricted to only the Laplace equation

using CUDA
using Images
using ImageView
using ColorSchemes
using Plots

# ============================================================
# GPU Walk-on-Spheres Laplace solver
#
# Self-contained: does NOT depend on Geometry.jl,
# BoundaryCondition.jl, Fields.jl, Render.jl, Images, etc.
#
# Boundary geometry supported:
#   - circles
#   - line segments
#
# Boundary conditions:
#   - constant scalar values
#   - spatially varying scalar values represented by
#     simple GPU-compatible analytic functions
#
# Public functions:
#   gpu_render
#   example_GPU
# ============================================================


# ============================================================
# Basic GPU-compatible geometry
# ============================================================

struct GPUPoint
    x::Float32
    y::Float32
end


# ============================================================
# GPU geometry representation
#
# Every boundary object is represented using arrays.
#
# geometry:
#   1 = circle
#   2 = line segment
#
# Circle:
#   x1,y1 = centre
#   x2    = radius
#
# Segment:
#   x1,y1 = first endpoint
#   x2,y2 = second endpoint
# ============================================================


const GPU_CIRCLE = Int32(1)
const GPU_SEGMENT = Int32(2)


# ============================================================
# Device-side distance routines
# ============================================================

@inline function distance_circle(
    x::Float32,
    y::Float32,
    cx::Float32,
    cy::Float32,
    r::Float32
)
    dx = x - cx
    dy = y - cy
    return abs(sqrt(dx * dx + dy * dy) - r)
end


@inline function distance_segment(
    x::Float32,
    y::Float32,
    x1::Float32,
    y1::Float32,
    x2::Float32,
    y2::Float32
)
    dx = x2 - x1
    dy = y2 - y1

    len2 = dx * dx + dy * dy

    if len2 <= 0.0f0
        ex = x - x1
        ey = y - y1
        return sqrt(ex * ex + ey * ey)
    end

    px = x - x1
    py = y - y1

    t = (px * dx + py * dy) / len2
    t = max(0.0f0, min(1.0f0, t))

    qx = x1 + t * dx
    qy = y1 + t * dy

    ex = x - qx
    ey = y - qy

    return sqrt(ex * ex + ey * ey)
end


@inline function geometry_distance(
    x::Float32,
    y::Float32,
    geometry::Int32,
    x1::Float32,
    y1::Float32,
    x2::Float32,
    y2::Float32
)
    if geometry == GPU_CIRCLE
        return distance_circle(x, y, x1, y1, x2)
    else
        return distance_segment(x, y, x1, y1, x2, y2)
    end
end


# ============================================================
# Find nearest boundary
#
# Returns:
#   minimum distance
#   index of nearest object
# ============================================================

@inline function nearest_geometry(
    x::Float32,
    y::Float32,
    geometry,
    x1,
    y1,
    x2,
    y2,
    n::Int32
)
    best_distance = Inf32
    best_index = Int32(1)

    @inbounds for k in Int32(1):n

        d = geometry_distance(
            x,
            y,
            geometry[k],
            x1[k],
            y1[k],
            x2[k],
            y2[k]
        )

        if d < best_distance
            best_distance = d
            best_index = k
        end
    end

    return best_distance, best_index
end


# ============================================================
# Boundary conditions
#
# bc_type:
#   1 = constant
#   2 = sinusoidal
#   3 = cosine
#   4 = radial
#
# bc_a, bc_b, bc_c are parameters.
# ============================================================

const BC_CONSTANT = Int32(1)
const BC_SINE = Int32(2)
const BC_COSINE = Int32(3)
const BC_RADIAL = Int32(4)


@inline function boundary_value(
    x::Float32,
    y::Float32,
    object_index::Int32,
    bc_type,
    bc_a,
    bc_b,
    bc_c
)

    t = bc_type[object_index]
    a = bc_a[object_index]
    b = bc_b[object_index]
    c = bc_c[object_index]

    if t == BC_CONSTANT

        return a

    elseif t == BC_SINE

        return a + b * sin(c * x)

    elseif t == BC_COSINE

        return a + b * cos(c * y)

    else
        # Radial boundary condition
        return a + b * sqrt(x * x + y * y) + c * sin(3.0f0 * x)
    end
end


# ============================================================
# One Walk-on-Spheres sample
#
# IMPORTANT:
# This function always returns Float32.
#
# There is deliberately no `nothing` return path.
# This avoids the previous
#
#   KernelError: kernel returns Union{Nothing,Float32}
#
# problem.
# ============================================================

@inline function walk_sample(
    x0::Float32,
    y0::Float32,

    geometry,
    x1,
    y1,
    x2,
    y2,

    bc_type,
    bc_a,
    bc_b,
    bc_c,

    n_objects::Int32,
    depth::Int32,
    epsilon::Float32,

    seed::UInt32
)

    x = x0
    y = y0

    # Simple per-thread pseudo-random state.
    state = seed

    @inbounds for step in Int32(1):depth

        dx, object_index = nearest_geometry(
            x,
            y,
            geometry,
            x1,
            y1,
            x2,
            y2,
            n_objects
        )

        if dx <= epsilon || step == depth

            # Move to nearest point.

            g = geometry[object_index]

            if g == GPU_CIRCLE

                cx = x1[object_index]
                cy = y1[object_index]
                r = x2[object_index]

                vx = x - cx
                vy = y - cy

                len = sqrt(vx * vx + vy * vy)

                if len > 0.0f0
                    bx = cx + r * vx / len
                    by = cy + r * vy / len
                else
                    bx = cx + r
                    by = cy
                end

                return boundary_value(
                    bx,
                    by,
                    object_index,
                    bc_type,
                    bc_a,
                    bc_b,
                    bc_c
                )

            else

                ax = x1[object_index]
                ay = y1[object_index]

                bx = x2[object_index]
                by = y2[object_index]

                vx = bx - ax
                vy = by - ay

                wx = x - ax
                wy = y - ay

                len2 = vx * vx + vy * vy

                t = (wx * vx + wy * vy) / len2

                t = max(0.0f0, min(1.0f0, t))

                px = ax + t * vx
                py = ay + t * vy

                return boundary_value(
                    px,
                    py,
                    object_index,
                    bc_type,
                    bc_a,
                    bc_b,
                    bc_c
                )
            end
        end

        # ----------------------------------------------------
        # Generate pseudo-random angle
        # ----------------------------------------------------

        state = state ⊻ (state << 13)
        state = state ⊻ (state >> 17)
        state = state ⊻ (state << 5)

        # Convert UInt32 to approximately [0,1].
        u = Float32(state % UInt32(1000000)) / 1000000.0f0

        θ = 2.0f0 * Float32(π) * u

        # Move to the boundary of the largest empty circle.
        x += dx * cos(θ)
        y += dx * sin(θ)
    end

    # This line is reachable only in pathological circumstances,
    # but guarantees a concrete Float32 return type.
    dx, object_index = nearest_geometry(
        x,
        y,
        geometry,
        x1,
        y1,
        x2,
        y2,
        n_objects
    )

    return boundary_value(
        x,
        y,
        object_index,
        bc_type,
        bc_a,
        bc_b,
        bc_c
    )
end


# ============================================================
# GPU render kernel
#
# One CUDA thread computes one image pixel.
# Each thread performs all Monte Carlo samples for that pixel.
# ============================================================

function renderKernel!(
    output,
    grid_x,
    grid_y,

    geometry,
    x1,
    y1,
    x2,
    y2,

    bc_type,
    bc_a,
    bc_b,
    bc_c,

    n_objects::Int32,
    samples::Int32,
    depth::Int32,
    nx::Int32,
    ny::Int32,
    epsilon::Float32
)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= nx && j <= ny

        x = grid_x[i]
        y = grid_y[j]

        total = 0.0f0

        # A deterministic seed unique to the pixel.
        base_seed =
            UInt32(i) * UInt32(1973) +
            UInt32(j) * UInt32(9277) +
            UInt32(i * j) * UInt32(26699) +
            UInt32(1)

        @inbounds for s in Int32(1):samples

            seed =
                base_seed +
                UInt32(s) * UInt32(7919)

            total += walk_sample(
                x,
                y,

                geometry,
                x1,
                y1,
                x2,
                y2,

                bc_type,
                bc_a,
                bc_b,
                bc_c,

                n_objects,
                depth,
                epsilon,

                seed
            )
        end

        @inbounds output[i, j] =
            total / Float32(samples)
    end

    # CUDA kernels must return nothing.
    return
end


# ============================================================
# GPU scene construction
# ============================================================

"""
Create the GPU representation of a scene.

Each geometry is encoded using four Float32 parameters:

Circle:
    x1 = centre x
    y1 = centre y
    x2 = radius
    y2 = unused

Segment:
    x1,y1 = first endpoint
    x2,y2 = second endpoint

Boundary condition parameters are stored separately.
"""
function make_gpu_scene(objects)

    n = length(objects)

    n > 0 || error("Scene must contain at least one object")

    geometry = Vector{Int32}(undef, n)

    x1 = Vector{Float32}(undef, n)
    y1 = Vector{Float32}(undef, n)
    x2 = Vector{Float32}(undef, n)
    y2 = Vector{Float32}(undef, n)

    bc_type = Vector{Int32}(undef, n)

    bc_a = Vector{Float32}(undef, n)
    bc_b = Vector{Float32}(undef, n)
    bc_c = Vector{Float32}(undef, n)

    for i in 1:n

        obj = objects[i]

        g = obj[1]

        # ----------------------------------------------------
        # Circle
        # ----------------------------------------------------

        if g[1] == :circle

            geometry[i] = GPU_CIRCLE

            x1[i] = Float32(g[2])
            y1[i] = Float32(g[3])
            x2[i] = Float32(g[4])
            y2[i] = 0.0f0

        # ----------------------------------------------------
        # Line segment
        # ----------------------------------------------------

        elseif g[1] == :segment

            geometry[i] = GPU_SEGMENT

            x1[i] = Float32(g[2])
            y1[i] = Float32(g[3])
            x2[i] = Float32(g[4])
            y2[i] = Float32(g[5])

        else
            error("Unknown GPU geometry")
        end

        # Boundary condition:
        #
        # (:constant, value)
        # (:sine, a, b, frequency)
        # (:cosine, a, b, frequency)
        # (:radial, a, b, c)

        bc = obj[2]

        if bc[1] == :constant

            bc_type[i] = BC_CONSTANT

            bc_a[i] = Float32(bc[2])
            bc_b[i] = 0.0f0
            bc_c[i] = 0.0f0

        elseif bc[1] == :sine

            bc_type[i] = BC_SINE

            bc_a[i] = Float32(bc[2])
            bc_b[i] = Float32(bc[3])
            bc_c[i] = Float32(bc[4])

        elseif bc[1] == :cosine

            bc_type[i] = BC_COSINE

            bc_a[i] = Float32(bc[2])
            bc_b[i] = Float32(bc[3])
            bc_c[i] = Float32(bc[4])

        elseif bc[1] == :radial

            bc_type[i] = BC_RADIAL

            bc_a[i] = Float32(bc[2])
            bc_b[i] = Float32(bc[3])
            bc_c[i] = Float32(bc[4])

        else
            error("Unknown GPU boundary condition")
        end
    end

    return (
        cu(geometry),
        cu(x1),
        cu(y1),
        cu(x2),
        cu(y2),
        cu(bc_type),
        cu(bc_a),
        cu(bc_b),
        cu(bc_c)
    )
end


# ============================================================
# GPU render
# ============================================================

"""
    gpu_render(
        xmin, xmax,
        ymin, ymax,
        nx, ny,
        objects;
        samples=100,
        depth=20,
        epsilon=1f-3
    )

Render a Laplace solution using GPU Walk-on-Spheres.

Returns a CPU Matrix{Float32}.
"""
function gpu_render(
    xmin::Real,
    xmax::Real,
    ymin::Real,
    ymax::Real,
    nx::Integer,
    ny::Integer,
    objects;
    samples::Integer = 100,
    depth::Integer = 20,
    epsilon::Real = 1e-3
)

    nx >= 2 || error("nx must be at least 2")
    ny >= 2 || error("ny must be at least 2")
    samples > 0 || error("samples must be positive")
    depth > 0 || error("depth must be positive")

    CUDA.functional() ||
        error("CUDA is not available")

    # --------------------------------------------------------
    # CPU coordinate vectors
    # --------------------------------------------------------

    xs = Float32[
        Float64(xmin) +
        (i - 1) * (Float64(xmax) - Float64(xmin)) / (nx - 1)
        for i in 1:nx
    ]

    ys = Float32[
        Float64(ymin) +
        (j - 1) * (Float64(ymax) - Float64(ymin)) / (ny - 1)
        for j in 1:ny
    ]

    # --------------------------------------------------------
    # Copy coordinates to GPU
    # --------------------------------------------------------

    gpu_x = cu(xs)
    gpu_y = cu(ys)

    output = CUDA.zeros(Float32, nx, ny)

    # --------------------------------------------------------
    # Copy scene data to GPU
    # --------------------------------------------------------

    (
        geometry,
        x1,
        y1,
        x2,
        y2,
        bc_type,
        bc_a,
        bc_b,
        bc_c
    ) = make_gpu_scene(objects)

    # --------------------------------------------------------
    # Launch kernel
    # --------------------------------------------------------

    threads = (16, 16)

    blocks = (
        cld(nx, threads[1]),
        cld(ny, threads[2])
    )

    @cuda threads=threads blocks=blocks renderKernel!(
        output,
        gpu_x,
        gpu_y,

        geometry,
        x1,
        y1,
        x2,
        y2,

        bc_type,
        bc_a,
        bc_b,
        bc_c,

        Int32(length(objects)),
        Int32(samples),
        Int32(depth),
        Int32(nx),
        Int32(ny),
        Float32(epsilon)
    )

    synchronize()

    return Array(output)
end


# ============================================================
# Convenience method using a simple domain tuple
# ============================================================

"""
    gpu_render(domain, objects; kwargs...)

where

    domain = (xmin, xmax, ymin, ymax)
"""
# ============================================================
# Convenience method using a simple domain tuple
# ============================================================

"""
    gpu_render(domain, objects;
               nx=256,
               ny=256,
               samples=100,
               depth=20,
               epsilon=1e-3)

where

    domain = (xmin, xmax, ymin, ymax)
"""
function gpu_render(
    domain::NTuple{4,<:Real},
    objects;
    nx::Integer = 256,
    ny::Integer = 256,
    samples::Integer = 100,
    depth::Integer = 20,
    epsilon::Real = 1e-3
)

    xmin, xmax, ymin, ymax = domain

    return gpu_render(
        xmin,
        xmax,
        ymin,
        ymax,
        nx,
        ny,
        objects;
        samples = samples,
        depth = depth,
        epsilon = epsilon
    )
end


# ============================================================
# Example
# ============================================================

"""
Run a complete GPU example.

The result is returned as a Matrix{Float32}.
"""

function example_GPU()

    # --------------------------------------------------------
    # Rendering domain
    # --------------------------------------------------------

    domain = (-3.0, 3.0, -3.0, 3.0)

    # --------------------------------------------------------
    # Boundary objects
    # --------------------------------------------------------

    objects = [

        (
            (:circle, 0.0, 0.0, 0.65),
            (:sine, 0.5, 0.45, 3.0)
        ),

        (
            (:circle, -1.35, 1.25, 0.55),
            (:constant, 1.0)
        ),

        (
            (:circle, 1.35, 1.25, 0.55),
            (:cosine, 0.5, 0.5, 3.0)
        ),

        (
            (:circle, -1.35, -1.25, 0.55),
            (:constant, 0.0)
        ),

        (
            (:circle, 1.35, -1.25, 0.55),
            (:radial, 0.0, 0.25, 0.35)
        ),

        (
            (:segment, -1.85, 0.0, 1.85, 0.0),
            (:sine, 0.5, 0.5, 5.0)
        )
    ]

    # --------------------------------------------------------
    # GPU render
    # --------------------------------------------------------

    rendering = gpu_render(
        domain,
        objects;
        nx = 256 * 2,
        ny = 256 * 2,
        samples = 64,
        depth = 20,
        epsilon = 1f-3
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------


    lo = minimum(rendering)
    hi = maximum(rendering)

    normalized =
        hi == lo ?
        zeros(Float32, size(rendering)) :
        (rendering .- lo) ./ (hi - lo)

    image = [
        get(ColorSchemes.viridis, normalized[i, j])
        for i in axes(normalized, 1), j in axes(normalized, 2)
    ]

    ImageView.imshow(image)

    return nothing
end

# ============================================================
# Animation helpers
# ============================================================

"""
Convert a rendering matrix to a Viridis RGB image.
"""
function rendering_image(rendering)

    lo = minimum(rendering)
    hi = maximum(rendering)

    normalized =
        hi == lo ?
        zeros(Float32, size(rendering)) :
        (rendering .- lo) ./ (hi - lo)

    return [
        get(ColorSchemes.viridis, normalized[i, j])
        for i in axes(normalized, 1), j in axes(normalized, 2)
    ]
end


"""
Display a rendering using Viridis.
"""
function display_rendering(rendering)

    image = rendering_image(rendering)

    ImageView.imshow(image)

    return nothing
end


# ============================================================
# Animated shapes
# ============================================================

"""
Animate the geometry while keeping the boundary conditions fixed.

The circles oscillate around their initial positions and the
line segment rotates.

The animation is intentionally low-resolution / low-sample so
that it can be used interactively.
"""
function example_GPU_animated_shapes(;
    frames = 60,
    nx = 800,
    ny = 800,
    samples = 60,
    depth = 30,
    epsilon = 1f-3
)

    domain = (-3.0, 3.0, -3.0, 3.0)

    # --------------------------------------------------------
    # Generate frames
    # --------------------------------------------------------

    animation = @animate for frame in 1:frames

        t = 2π * (frame - 1) / frames

        # Oscillating circle positions
        c1x = -1.35 + 0.35 * sin(t)
        c1y =  1.25 + 0.20 * cos(t)

        c2x =  1.35 + 0.30 * cos(t)
        c2y =  1.25 + 0.25 * sin(t)

        c3x = -1.35 + 0.25 * cos(t)
        c3y = -1.25 + 0.30 * sin(t)

        c4x =  1.35 + 0.30 * sin(t)
        c4y = -1.25 + 0.20 * cos(t)

        # Rotating line segment
        θ = 0.35 * sin(t)

        half_length = 1.85

        x1 = -half_length * cos(θ)
        y1 = -half_length * sin(θ)

        x2 =  half_length * cos(θ)
        y2 =  half_length * sin(θ)

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        objects = [

            (
                (:circle, 0.0, 0.0, 0.65),
                (:sine, 0.5, 0.45, 3.0)
            ),

            (
                (:circle, c1x, c1y, 0.55),
                (:constant, 1.0)
            ),

            (
                (:circle, c2x, c2y, 0.55),
                (:cosine, 0.5, 0.5, 3.0)
            ),

            (
                (:circle, c3x, c3y, 0.55),
                (:constant, 0.0)
            ),

            (
                (:circle, c4x, c4y, 0.55),
                (:radial, 0.0, 0.25, 0.35)
            ),

            (
                (:segment, x1, y1, x2, y2),
                (:sine, 0.5, 0.5, 5.0)
            )
        ]

        rendering = gpu_render(
            domain,
            objects;
            nx = nx,
            ny = ny,
            samples = samples,
            depth = depth,
            epsilon = epsilon
        )

        # Plot rather than ImageView so that frames can be
        # collected into an animation.
        heatmap(
            rendering',
            c = :magma,
            aspect_ratio = 1,
            clims = (0, 1),
            axis = false,
            legend = false,
        )
    end

    # --------------------------------------------------------
    # Save and display animation
    # --------------------------------------------------------

    gif(animation, "GPU_animated_shapes.gif", fps = 15)

    println("Saved GPU_animated_shapes.gif")

    return nothing
end


# ============================================================
# Animated boundary conditions
# ============================================================

"""
Animate the boundary conditions while keeping the geometry fixed.

The boundary values oscillate continuously in time.
"""
function example_GPU_animated_boundary_conditions(;
    frames = 60,
    nx = 500,
    ny = 500,
    samples = 32,
    depth = 15,
    epsilon = 1f-3
)

    domain = (-3.0, 3.0, -3.0, 3.0)

    # --------------------------------------------------------
    # Fixed geometry
    # --------------------------------------------------------

    animation = @animate for frame in 1:frames

        t = 2π * (frame - 1) / frames

        # ----------------------------------------------------
        # Time-dependent boundary-condition parameters
        # ----------------------------------------------------

        central_offset =
            0.5 + 0.35 * sin(t)

        central_amplitude =
            0.25 + 0.20 * cos(t)

        upper_right_offset =
            0.5 + 0.30 * cos(t)

        lower_right_amplitude =
            0.25 + 0.20 * sin(t)

        line_offset =
            0.5 + 0.30 * sin(t + π / 3)

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        objects = [

            (
                (:circle, 0.0, 0.0, 0.65),
                (
                    :sine,
                    central_offset,
                    central_amplitude,
                    3.0
                )
            ),

            (
                (:circle, -1.35, 1.25, 0.55),
                (
                    :constant,
                    0.5 + 0.5 * sin(t)
                )
            ),

            (
                (:circle, 1.35, 1.25, 0.55),
                (
                    :cosine,
                    upper_right_offset,
                    0.4,
                    3.0
                )
            ),

            (
                (:circle, -1.35, -1.25, 0.55),
                (
                    :constant,
                    0.5 + 0.5 * cos(t)
                )
            ),

            (
                (:circle, 1.35, -1.25, 0.55),
                (
                    :radial,
                    0.0,
                    lower_right_amplitude,
                    0.35
                )
            ),

            (
                (:segment, -1.85, 0.0, 1.85, 0.0),
                (
                    :sine,
                    line_offset,
                    0.4,
                    5.0
                )
            )
        ]

        rendering = gpu_render(
            domain,
            objects;
            nx = nx,
            ny = ny,
            samples = samples,
            depth = depth,
            epsilon = epsilon
        )

        heatmap(
            rendering',
            c = :inferno,
            aspect_ratio = 1,
            clims = (0, 1),
            axis = false,
            legend = false,
        )
    end

    gif(
        animation,
        "GPU_animated_boundary_conditions.gif",
        fps = 15
    )

    println(
        "Saved GPU_animated_boundary_conditions.gif"
    )

    return nothing
end
