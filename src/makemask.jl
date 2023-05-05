@inline function lap_u(u,ix,iy,dx,dy,Nx,Ny)
    ## Laplacian Stencil for the ODE 
    left = ix > 1 ? u[ix-1,iy] : u[ix+1,iy]
    right = ix < Nx ? u[ix+1,iy] : u[ix-1,iy]
    bottom = iy > 1 ? u[ix,iy-1] : u[ix,iy+1]
    top = iy < Ny ? u[ix,iy+1] : u[ix,iy-1]

    return ((right + left - 2.0*u[ix,iy])/dx^2 + (top + bottom - 2.0*u[ix,iy])/dy^2)
end

function shape_smooth(du,u,p,t)
    # RXN diffusion equation
    zeta,Nx,Ny,dx,dy = p
    @inline function rxn(i,j)
        return (2.0*u[i,j]-1.0)*(-1.0+(2.0*u[i,j]-1.0)^2)
    end
    @views @inbounds for I in CartesianIndices((Nx, Ny))
        i,j= Tuple(I);
        du[i,j] = zeta^2*lap_u(u,i,j,dx,dy,Nx,Ny) - rxn(i,j);
    end
end


function makeMask(filename::String; imgthresh=0.1,tf = 0.1,filltol=1e-3,ζ=0.04)
    ψ0 = makePngMask(filename,imgthresh)
    Nx=size(ψ0,1)
    Ny=size(ψ0,2)
    x = LinRange(0.0,1,Nx)
    y= LinRange(0.0,1,Ny)
    dx = x[2]-x[1];dy=x[2]-x[1];
    p = (ζ,Nx,Ny,dx,dy)
    ψ0 =ψ0[end:-1:1,:]
    tspan = (0.0,tf);
    prob = ODEProblem(shape_smooth, ψ0, tspan, p);
    sol = solve(prob,CVODE_BDF(linear_solver=:GMRES),abstol=1e-10,reltol=1e-10);

    ψ = sol.u[end];

    ψ[ψ.<filltol] .= tol
    ψ[ψ.>0.99] .= 1.0

    ψ_binary = copy(ψ)
    ψ_binary[ψ_binary.<0.5] .= 0.0
    ψ_binary[ψ_binary.>=0.5] .= 1.0

    return ψ,ψ_binary
end

function makePngMask(filename::String, threshold::Real = 0.5; invert = false)
    # Read the image
    img = load(filename)
    
    # Convert the image to grayscale
    img_gray = Gray.(img)
    
    # Allocate an empty binary array of the same size as the image
    binary_array = falses(size(img_gray))
    
    # Iterate through each pixel and set the corresponding binary value
    for i in eachindex(img_gray)
        binary_array[i] = img_gray[i].val > threshold ? true : false
    end
    if invert
        binary_array = .~binary_array
    end
    return Float64.(binary_array)
end