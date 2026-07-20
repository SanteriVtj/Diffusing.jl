abstract type AbstractDiffusionSDE end

struct VariancePreservingSDE <: AbstractDiffusionSDE
    β_min::Float64
    β_max::Float64
    β_schedule::Function
end

drift(sde::VariancePreservingSDE, x, t) = -0.5*sde.β_schedule(t).*x
diffusion(sde::VariancePreservingSDE, x, t) = sqrt(sde.β_schedule(t))

struct VarianceExplodingSDE <: AbstractDiffusionSDE
    σ_min::Float64
    σ_max::Float64
end

drift(sde::VarianceExplodingSDE, x, t) = zeros(eltype(x), size(x))
diffusion(sde::VarianceExplodingSDE, x, t) = sde.σ_min * (sde.σ_max/sde.σ_max)^t*log(sde.σ_max/sde.σ_min)
