# Parameter Setup
function setup_CH(ψ; gpuflag = false,dampcoeff=1.0,kw...)
    #Set up the simulation domain from the mask
    Nx, Ny = size(ψ) # grab size
    x = LinRange(0.0, 1, Nx) # x domain
    dx = x[2] - x[1];
    y = LinRange(0.0, 1, Ny) # y domain
    dy = y[2] - y[1];

    # Set up Laplacian stencils
    ∇2x = Tridiagonal([1.0 for i in 1:Nx-1],[-2.0 for i in 1:Nx],[1.0 for i in 1:Nx-1])
    ∇2x[1,2] = 2.0
    ∇2x[end,end-1] = 2.0
    ∇2y = Tridiagonal([1.0 for i in 1:Ny-1],[-2.0 for i in 1:Ny],[1.0 for i in 1:Ny-1])
    ∇2y[1,2] = 2.0
    ∇2y[end,end-1] = 2.0
    ∇2y = ∇2y'

    #Set up d/dx stencils
    ∇x= Tridiagonal([-1.0 for i in 1:Nx-1],[0.0 for i in 1:Nx],[1.0 for i in 1:Nx-1]);
    ∇x[1,2]=0.0
    ∇x[end,end-1]=0.0

    #Set up d/dy stencils
    ∇y= Tridiagonal([-1.0 for i in 1:Ny-1],[0.0 for i in 1:Ny],[1.0 for i in 1:Ny-1]);
    ∇y[1,2]=0.0
    ∇y[end,end-1]=0.0
    ∇y =∇y'

    # Normalize stencils by appropriate differentials
    ∇2x ./= dx^2;
    ∇2y ./= dy^2;
    ∇x ./= 2*dx;
    ∇y ./= 2*dy;

    #Precompute mask gradients
    damp_factor = dampcoeff*norm(ψ);

    ∇ψ_x = ∇x*ψ./damp_factor; 
    ∇ψ_y = ψ*∇y./damp_factor;

    if ~gpuflag
        # Set up the caches
        ∇c_x=zeros(Nx,Ny); ∇c_x_c = DiffCache(∇c_x ;kw...);
        ∇c_y=zeros(Nx,Ny); ∇c_y_c = DiffCache(∇c_y ;kw...);
        ∇2c=zeros(Nx,Ny); ∇2c_c = DiffCache(∇2c ;kw...)
        μ = zeros(Nx,Ny); μ_c = DiffCache(μ ;kw...)
        ∇2μ=zeros(Nx,Ny); ∇2μ_c = DiffCache(∇2μ ;kw...)
        ∇μ_x=zeros(Nx,Ny); ∇μ_x_c = DiffCache(∇μ_x ;kw...)
        ∇μ_y=zeros(Nx,Ny); ∇μ_y_c = DiffCache(∇μ_y ;kw...)
        return x,y,(du,u,p,t)->CHCacheFuncCPU(du,u,p,t,ψ,∇ψ_x,∇ψ_y,∇x,∇y,∇2x,∇2y,∇c_x_c,∇c_y_c,∇2c_c,μ_c,∇2μ_c,∇μ_x_c,∇μ_y_c)
    else
        # Convert everything to Float32 and send to the gpu
        ψ_g = CuArray(Float32.(ψ))
        ∇x_g = CuArray(Float32.(∇x))
        ∇y_g = CuArray(Float32.(∇y))
        ∇2x_g = CuArray(Float32.(∇2x))
        ∇2y_g = CuArray(Float32.(∇2y))
        ∇ψ_x_g = CuArray(Float32.(∇ψ_x))
        ∇ψ_y_g = CuArray(Float32.(∇ψ_y))
        ∇c_x_g = CuArray(Float32.(∇c_x))
        ∇c_y_g = CuArray(Float32.(∇c_y))
        ∇2c_g = CuArray(Float32.(∇2c))
        μ_g = CuArray(Float32.(μ))
        ∇2μ_g = CuArray(Float32.(∇2μ))
        ∇μ_x_g = CuArray(Float32.(∇μ_x))
        ∇μ_y_g = CuArray(Float32.(∇μ_y))
        return x,y,(du,u,p,t)->CHCacheFuncGPU(du,u,p,t,ψ_g,∇ψ_x_g,∇ψ_y_g,∇x_g,∇y_g,∇2x_g,∇2y_g,∇c_x_g,∇c_y_g,∇2c_g,μ_g,∇2μ_g,∇μ_x_g,∇μ_y_g)
    end

end
function genC0(Nx,Ny,ψ_b;noise_fac=0.05)
    c0out = zeros(Nx,Ny)
    for i=1:Nx,j=1:Ny
        c0out[i,j]= ψ_b[i,j] ≈ 1.0 ? 0.5 + noise_fac*(2.0.*rand().-1.0) : 0.5 + 0.001*noise_fac*(2.0.*rand().-1.0);
    end
    return c0out;
end
## ODE solver set up 
function makesparseprob(rhsfunc,c0,tspan,p)
    dc0=similar(c0);
    jac_sparsity = Symbolics.jacobian_sparsity((du, u) ->rhsfunc(du,u,p,0),dc0,c0);
    colorvec = matrix_colors(jac_sparsity);
    f = ODEFunction(rhsfunc;jac_prototype=jac_sparsity,colorvec=colorvec);
    sparse_prob = ODEProblem(f,c0,tspan,p);
    return sparse_prob
end
## Solution Handling Utility Functions
struct CHsol
    sol
    ψ_binary
end
function heatgif(A::AbstractArray{<:Number,3}; kwargs...)
    p = heatmap(zeros(size(A, 1), size(A, 2)); kwargs...)
    anim = @animate for i=1:size(A,3)
        heatmap!(p[1], @view A[:,:,i])
    end
    return anim
end
function makegif(fullsol::CHsol; fpsv=10)
    masksol = Array(fullsol.sol.u).*fullsol.ψ_binary;
    anim = heatgif(Array(masksol))
    return gif(anim, fps = fpsv)
end
function plotlast(fullsol::CHsol;kw...)
    tmp =Array((fullsol.sol).u[end]).*fullsol.ψ_binary
    heatmap(tmp;kw...)
end