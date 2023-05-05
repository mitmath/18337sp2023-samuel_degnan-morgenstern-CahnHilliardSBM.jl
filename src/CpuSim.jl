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
    x,y,rhsfunc =setup_CH(ψ; gpuflag = false,kw...)
    tspan = (simargs.t0,simargs.tf);
    prob = makesparseprob(rhsfunc,simargs.c0,tspan,simargs.p)
    if saveflag
        sol = solve(prob,CVODE_BDF(linear_solver=:GMRES),saveat=range(simargs.t0,simargs.tf,length=simargs.nt))
    else
        sol = solve(prob,CVODE_BDF(linear_solver=:GMRES),save_everystep=false);
    end
    return CHsol(sol,SimToken.ψ_binary)
end
