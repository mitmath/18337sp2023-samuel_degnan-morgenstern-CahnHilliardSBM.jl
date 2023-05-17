using Optimization, DifferentialEquations, Plots, Random,Optim,OptimizationOptimJL,OptimizationPolyalgorithms
using LinearAlgebra,SparseArrays,PreallocationTools,SciMLSensitivity,Symbolics,SparseDiffTools,Sundials,LinearSolve
using ForwardDiff,DelimitedFiles
c0 = readdlm("goodc0.csv")
datasize = 100;
tspan = (0.0f0, 5.0f0)
tsteps = collect(range(tspan[1], tspan[2], length = datasize))
function GCH_2D_mul_full2(du, u, p, t,ψ,∇x,∇y,∇2x,∇2y,∇ψ_x,∇ψ_y,∇c_x,∇c_y,∇2c,μ,∇2μ,∇μ_x,∇μ_y)
    D, Ω=p
    c = @view u[:,:]
    dc = @view du[:,:]
    
    #Set up caches from Lazy Buffer Caches
    ∇c_x_t = get_tmp(∇c_x,u)
    ∇c_y_t = get_tmp(∇c_y,u)
    ∇2c_t = get_tmp(∇2c,u)
    μ_t = get_tmp(μ,u)
    ∇2μ_t = get_tmp(∇2μ,u)
    ∇μ_x_t = get_tmp(∇μ_x,u)
    ∇μ_y_t = get_tmp(∇μ_y,u)
    
    #Compute ∇c
    mul!(∇c_x_t,∇x,c) # Compute (∇c)ₓ = ∇x*c
    mul!(∇c_y_t,c,∇y) # Compute (∇c)_y = c*∇y
    
    #Compute ∇2c
    mul!(∇2c_t,∇2x,c) # Compute (∇2c)ₓ = c*∇2x
    mul!(∇2c_t,c,∇2y,1.0,1.0) #∇2c = 1*(∇2c)ₓ + 1*(∇2y)*c

    @. μ_t = log(max(1e-10,c./(1.0 - c)))+ Ω*(1.0 - 2.0*c) .- 0.001*((∇c_x_t*∇ψ_x  + ∇c_y_t*∇ψ_y)./ψ + ∇2c_t); # κ=0.001

    #Compute ∇2μ
    mul!(∇2μ_t,∇2x,μ_t) # Compute (∇2μ)ₓ = μ*∇2x
    mul!(∇2μ_t,μ_t,∇2y,1.0,1.0) #∇2μ = 1*(∇2μ)ₓ + 1*(∇2y)*μ
    #Compute ∇μ
    mul!(∇μ_x_t,∇x,μ_t) # Compute (∇μ)ₓ = ∇x*μ
    mul!(∇μ_y_t,μ_t,∇y) # Compute (∇μ)_y = μ*∇y
    @. dc = D*(c*(1.0-c)*((∇ψ_x*∇μ_x_t + ∇ψ_y*∇μ_y_t)./ψ + ∇2μ_t) + (1.0-2.0*c)*(∇c_x_t*∇μ_x_t + ∇c_y_t*∇μ_y_t))
    return nothing
end
function loss_proto(θ,prob,tsteps,ψ_binary,ode_data)
    tmp_prob = remake(prob, p = θ)
    tmp_sol = solve(tmp_prob, TRBDF2(), saveat =tsteps,abstol=1e-8,reltol=1e-8);
    #if tmp_sol.retcode == ReturnCode.Success
    if size(tmp_sol) == size(ode_data)
        return sum(abs2, (ψ_binary.*(Array(tmp_sol) - ode_data)))
    else
        return Inf
    end
end

ψ = readdlm("psi.csv"); ψ_binary = readdlm("psi_b.csv")
ψ = ψ[end:-1:1, :];ψ_binary = ψ_binary[end:-1:1, :]

Nx, Ny = size(ψ)
x = LinRange(0.0, 1, Nx)
y = LinRange(0.0, 1, Ny)
dx = x[2] - x[1]
dy = y[2] - y[1]

D = 0.1
κ = 0.001
Ω = 3.0

p = [D, Ω];

∇2x = Tridiagonal([1.0 for i in 1:Nx-1],[-2.0 for i in 1:Nx],[1.0 for i in 1:Nx-1])
∇2x[1,2] = 2.0
∇2x[end,end-1] = 2.0
∇2y= deepcopy(∇2x)
∇2y = ∇2y'

∇x= Tridiagonal([-1.0 for i in 1:Nx-1],[0.0 for i in 1:Nx],[1.0 for i in 1:Nx-1]);
∇x[1,2]=0.0
∇x[end,end-1]=0.0

∇y= Tridiagonal([-1.0 for i in 1:Ny-1],[0.0 for i in 1:Ny],[1.0 for i in 1:Ny-1]);
∇y[1,2]=0.0
∇y[end,end-1]=0.0
∇y =∇y'


∇2x ./= dx^2;
∇2y ./= dy^2;
∇x ./= 2*dx;
∇y ./= 2*dy;


∇ψ_x = ∇x*ψ 
∇ψ_y = ψ*∇y

∇c_x=zeros(Nx,Ny);
∇c_y=zeros(Nx,Ny);
∇2c=zeros(Nx,Ny);
μ = zeros(Nx,Ny);
∇2μ=zeros(Nx,Ny);
∇μ_x=zeros(Nx,Ny);
∇μ_y=zeros(Nx,Ny);

chunk_size = 25;

∇c_x = DiffCache(∇c_x,chunk_size,levels=3)
∇c_y = DiffCache(∇c_y,chunk_size,levels=3)
∇2c = DiffCache(∇2c,chunk_size,levels=3)
μ = DiffCache(μ,chunk_size,levels=3)
∇2μ = DiffCache(∇2μ,chunk_size,levels=3)
∇μ_x = DiffCache(∇μ_x,chunk_size,levels=3)
∇μ_y = DiffCache(∇μ_y,chunk_size,levels=3)

dc0 = similar(c0)

GCH_2D_mul!(du,u,p,t) = GCH_2D_mul_full2(du, u, p, t,ψ,∇x,∇y,∇2x,∇2y,∇ψ_x,∇ψ_y,∇c_x,∇c_y,∇2c,μ,∇2μ,∇μ_x,∇μ_y)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> GCH_2D_mul!(du,u,p,0),dc0,c0);
colorvec = matrix_colors(jac_sparsity);
f = ODEFunction(GCH_2D_mul!;jac_prototype=jac_sparsity,colorvec=colorvec);
prob1 = ODEProblem(f,c0,tspan,p);
ode_first_sol = solve(prob1, TRBDF2(), saveat = tsteps);

prob = remake(prob1,u0=ode_first_sol.u[1]);
ode_data = Array(solve(prob, TRBDF2(), saveat = tsteps,abstol=1e-8,reltol=1e-8));

loss(θ)=loss_proto(θ,prob,tsteps,ψ_binary,ode_data)

pinit = [0.3,2.5]; 
loss(pinit)

optfun = OptimizationFunction((u,_)->loss(u), Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optfun, pinit)

iter = 0
iter_track=[]
loss_track = []
callback = function(p, l)
    global  iter,iter_track,loss_track
    iter += 1
    if l !== Inf
        push!(iter_track,iter)
        push!(loss_track,l)
    end
    display("Iteration $(iter), loss = $(l)")
    return false
end
@time optsol = solve(optprob, NewtonTrustRegion(),callback=callback,maxiters=100,abstol=1e-4);
lplt=plot(iter_track,loss_track,yaxis=:log,legend=false,xlabel="Iterations",ylabel="L2 Loss",grid=false,marker=:dot,markercolor=:gray,color=:maroon)
savefig(lplt,"Loss_plot.png")