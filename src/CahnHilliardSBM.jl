###############################################
# Local Working Directory Management
###############################################
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Get the absolute path of the current script file
script_path = abspath(@__FILE__)

# Get the directory containing the script file
script_dir = dirname(script_path)

# Set the working directory to the script directory
cd(script_dir)

joinpath(@__DIR__, "relative", "path", "to", "\\masks")
joinpath(@__DIR__, "relative", "path", "to", "\\initcond")
###############################################
# CahnHilliardSBM.jl Set Up
###############################################
using PreallocationTools,LinearAlgebra,CUDA
using Symbolics,SparseDiffTools,SparseArrays,LinearAlgebra,CUDA,PreallocationTools,Plots,DifferentialEquations
using Sundials
using DelimitedFiles,Images
using Optimization,ForwardDiff,Optim,OptimizationOptimJL,SciMLSensitivity
include("RHSFunc.jl")
export CHCacheFuncCPU,CHCacheFuncGPU
include("Utils.jl")
export setup_CH,makesparseprob,makegif,genC0
include("CpuSim.jl")
export SimTokenCPU,runsimCPU
include("GpuSim.jl")
export SimTokenGPU,runsimGPU
include("MakeMask.jl")
export makeMask,makePngMask
include("ParamEst.jl")
export run_pe