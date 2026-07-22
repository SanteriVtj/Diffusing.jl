struct GaussianFourierProjection{F} <: AbstractLuxLayer
    embed_dim::Int
    scale::F
end

function GaussianFourierProjection(embed_dim::Int, scale::Float32=30.0f0)
    return GaussianFourierProjection{typeof(scale)}(embed_dim, scale)
end

function Lux.initialstates(rng::AbstractRNG, l::GaussianFourierProjection)
    W = randn(rng, Float32, l.embed_dim÷2)*l.scale
    return (W = W,)
end

Lux.initialparameters(::AbstractRNG, ::GaussianFourierProjection) = NamedTuple()

function (l::GaussianFourierProjection)(t, ps, st)
    t_arr = t isa AbstractArray ? t : [t]
    W = st.W
    t_row = reshape(t_arr, 1, length(t_arr))

    proj = (2.0f0*Float32(π)*(W.*t_row))
    emb = cat(sin.(proj), cos.(proj); dims=1)

    return emb, st
end
