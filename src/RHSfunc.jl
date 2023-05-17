function CHCacheFuncCPU(du,u,p,t,ψ,∇x,∇y,∇2x,∇2y,∇ψ_x,∇ψ_y,∇c_x,∇c_y,∇2c,μ,∇2μ,∇μ_x,∇μ_y)
    D, κ, Ω=p
    c = @view u[:,:]
    dc = @view du[:,:]
    
    #Set up caches from DiffCache
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

    @. μ_t = log(max(1e-10,c./(1.0 - c)))+ Ω*(1.0 - 2.0*c) .- κ*((∇c_x_t*∇ψ_x  + ∇c_y_t*∇ψ_y)./ψ + ∇2c_t);

    #Compute ∇2μ
    mul!(∇2μ_t,∇2x,μ_t) # Compute (∇2μ)ₓ = μ*∇2x
    mul!(∇2μ_t,μ_t,∇2y,1.0,1.0) #∇2μ = 1*(∇2μ)ₓ + 1*(∇2y)*μ
    #Compute ∇μ
    mul!(∇μ_x_t,∇x,μ_t) # Compute (∇μ)ₓ = ∇x*μ
    mul!(∇μ_y_t,μ_t,∇y) # Compute (∇μ)_y = μ*∇y
    @. dc = D*(c*(1.0-c)*((∇ψ_x*∇μ_x_t + ∇ψ_y*∇μ_y_t)./ψ + ∇2μ_t) + (1.0-2.0*c)*(∇c_x_t*∇μ_x_t + ∇c_y_t*∇μ_y_t))
    return nothing
end
function CHCacheFuncGPU(du,u,p,t,ψ,∇x,∇y,∇2x,∇2y,∇ψ_x,∇ψ_y,∇c_x,∇c_y,∇2c,μ,∇2μ,∇μ_x,∇μ_y)
    D, κ, Ω =p
    c = @view u[:,:]
    dc = @view du[:,:]
    
    #Compute ∇c
    mul!(∇c_x,∇x,c) # Compute (∇c)ₓ = ∇x*c
    mul!(∇c_y,c,∇y) # Compute (∇c)_y = c*∇y
    
    #Compute ∇2c
    mul!(∇2c,∇2x,c) # Compute (∇2c)ₓ = c*∇2x
    mul!(∇2c,c,∇2y,1.0,1.0) #∇2c = 1*(∇2c)ₓ + 1*(∇2y)*c

    #Compute Chemical Potential
    @. μ_t = log(max(1e-10,c./(1.0 - c)))+ Ω*(1.0 - 2.0*c) .- κ*((∇c_x*∇ψ_x  + ∇c_y*∇ψ_y)./ψ + ∇2c);

    #Compute ∇2μ
    mul!(∇2μ,∇2x,μ) # Compute (∇2μ)ₓ = μ*∇2x
    mul!(∇2μ,μ,∇2y,1.0,1.0) #∇2μ = 1*(∇2μ)ₓ + 1*(∇2y)*μ
    #Compute ∇μ
    mul!(∇μ_x,∇x,μ) # Compute (∇μ)ₓ = ∇x*μ
    mul!(∇μ_y,μ,∇y) # Compute (∇μ)_y = μ*∇y
    @. dc = D*(c*(1.0-c)*((∇ψ_x*∇μ_x + ∇ψ_y*∇μ_y)./ψ + ∇2μ) + (1.0-2.0*c)*(∇c_x*∇μ_x + ∇c_y*∇μ_y))
    return nothing
end