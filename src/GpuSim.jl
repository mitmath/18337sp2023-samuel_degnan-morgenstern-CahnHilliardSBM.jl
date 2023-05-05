module gpu_sim


export SimTokenGPU,runsimGPU

using .Utils:setup_CH,makesparseprob,CHsol
using DifferentialEquations,CUDA

struct SimTokenGPU
    ψ
    ψ_binary
    c0
    t0
    tf
    p
    nt
end
function runsimGPU(simargs::SimTokenGPU;saveflag =false, kw...)
    x,y,rhsfunc =setup_CH(ψ; gpuflag = true,kw...)
    tspan = (simargs.t0,simargs.tf);
    prob = makesparseprob(rhsfunc,simargs.c0,tspan,simargs.p)
    if saveflag
        sol = solve(prob,ROCK2(),saveat=range(simargs.t0,simargs.tf,length=simargs.nt))
    else
        sol = solve(prob,ROCK2(),save_everystep=false);
    end
    return CHsol(Array(sol),SimToken.ψ_binary)
end

end