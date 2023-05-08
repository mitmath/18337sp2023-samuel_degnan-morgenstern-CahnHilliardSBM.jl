#include("Utils.jl")
struct SimTokenCPU
    ψ
    ψ_binary
    c0
    t0
    tf
    p
    nt
end
function runsimCPU(simargs::SimTokenCPU;saveflag =false, kw...)
    x,y,rhsfunc =setup_CH(simargs.ψ; gpuflag = false,kw...)
    tspan = (simargs.t0,simargs.tf);
    prob = makesparseprob(rhsfunc,simargs.c0,tspan,simargs.p)
    if saveflag
        sol = solve(prob,CVODE_BDF(linear_solver=:GMRES),saveat=range(simargs.t0,simargs.tf,length=simargs.nt),abstol=1e-10,reltol=1e-10)
    else
        sol = solve(prob,CVODE_BDF(linear_solver=:GMRES),save_everystep=false,abstol=1e-10,reltol=1e-10);
    end
    return CHsol(sol,simargs.ψ_binary)
end
