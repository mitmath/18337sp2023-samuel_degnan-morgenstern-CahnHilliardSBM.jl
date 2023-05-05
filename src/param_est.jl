module param_est
using DifferentialEquations,Optimization,ForwardDiff,PreallocationTools,Optim,OptimizationOptimJL,LinearAlgebra,SciMLSensitivity
import runsim,rhsfunc,utils
export run_pe
function proto_loss(θ,prob,tsteps,ode_data,ψ_binary)
    tmp_prob = remake(prob, p = θ)
    tmp_sol = solve(tmp_prob, TRBDF2(), saveat =tsteps,sensealg =ForwardDiffSensitivity())
    #if tmp_sol.retcode == ReturnCode.Success
    if size(tmp_sol) == size(ode_data)
        return sum(abs2, (ψ_binary.*(Array(tmp_sol) - ode_data)))
    else
        return Inf
    end
end

function set_pe(ψ,c0_messy,ptruth,tspan,Nsteps; tcleaning=1e-4)
    tsteps = collect(range(tspan[1], tspan[2], length = Nsteps))
    x,y,rhsfunc =setup_CH(ψ; gpuflag = false,levels=3)
    prob = makesparseprob(rhsfunc,c0,(0,tcleaning),ptruth)
    tmpsol=solve(prob,TRBDF2(),save_everystep=false);
    newc0 =tmpsol.u[end];
    prob = remake(prob,tspan=tspan,c0=newc0);
    ode_data = Array(solve(prob, TRBDF2(), saveat = tsteps));
    return  tsteps,ode_data,prob
end
function callback(p, l)
    global  iter
    iter += 1
    display("Iteration $(iter), loss = $(l)")
    return false
end
function solve_pe(pinit,loss;iter_max=200)
    optfun = OptimizationFunction((u,_)->loss(u), Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfun, pinit)
    @time optsol = solve(optprob, Optim.NewtonTrustRegion(),callback=callback;maxiters=iter_max)
    return optsol
end


function run_pe(ψ,ψ_binary,c0_messy,ptruth;tspan=(0.0,1.0),Nsteps=100,itmax=200;)
    tsteps,ode_data,prob=set_pe(ψ,c0_messy,ptruth,tspan,Nsteps)
    loss(θ) = proto_loss(θ,prob,tsteps,ode_data,ψ_binary)
    optsol=solve_pe(pinit,loss;iter_max=itmax)
    return optsol
end

end