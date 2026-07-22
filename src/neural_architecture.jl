# Down and upsampling blocks

struct DownBlock{C,P,T} <: AbstractLuxContainerLayer{(:conv,:pool,:time_embedding)}
    conv::C
    pool::P
    time_embedding::T
end

function DownBlock(in_chs::Int, out_chs::Int, embed_dims::Int)
    conv = Chain(
                 Conv((3,3), in_chs=>out_chs, relu; pad=SamePad()),
                 Conv((3,3), out_chs=>out_chs, relu; pad=SamePad())
                )
    pool = MaxPool((2,2))
    time_embedding = Lux.Dense(embed_dims=>out_chs, relu)
    return DownBlock(conv, pool, time_embedding)
end

function (d::DownBlock)((x,t_embed), ps, st)
    t_proj, st_t = d.time_embedding(t_embed, ps.time_embedding, st.time_embedding)
    features, st_conv = d.conv(x, ps.conv, st.conv)
    t_reshaped = reshape(t_proj, 1, 1, size(t_proj,1), size(t_proj,2))
    features_conditioned = features.+t_reshaped
    downsampled, st_pool = d.pool(features_conditioned, ps.pool, st.pool)
    return (features_conditioned, downsampled), (conv = st_conv, pool=st_pool, time_embedding=st_t)
end

struct UpBlock{U,C,T} <: Lux.AbstractLuxContainerLayer{(:upsample, :conv, :time_embedding)}
    upsample::U
    conv::C
    time_embedding::T
end

function UpBlock(in_chs::Int, out_chs::Int, embed_dim::Int)
    upsample = Lux.Upsample(:bilinear, scale=2)
    conv = Chain(
                 Conv((3,3), (in_chs + out_chs)=>out_chs, relu; pad=SamePad()),
                 Conv((3,3), out_chs=>out_chs, relu; pad=SamePad())
                )
    time_embedding = Lux.Dense(embed_dim=>out_chs, relu)
    return UpBlock(upsample, conv, time_embedding)
end

function (u::UpBlock)((x, skip, t_embed), ps, st)
    x_up, st_up = u.upsample(x, ps.upsample, st.upsample)
    x_concat = cat(x_up, skip; dims=3)
    out, st_conv = u.conv(x_concat, ps.conv, st.conv)

    t_proj, st_t = u.time_embedding(t_embed, ps.time_embedding, st.time_embedding)
    t_reshaped = reshape(t_proj, 1, 1, size(t_proj, 1), size(t_proj, 2))
    out_conditioned = out .+ t_reshaped
    return out_conditioned, (upsample=st_up, conv=st_conv, time_embedding=st_t)
end

# UNet definition

struct UNet <: Lux.AbstractLuxContainerLayer{(:gauss_embed, :time_mlp, :down1, :down2, :bottleneck, :up2, :up1, :final_conv)}
    gauss_embed::GaussianFourierProjection
    time_mlp::Chain
    down1::DownBlock
    down2::DownBlock
    bottleneck::Chain
    up2::UpBlock
    up1::UpBlock
    final_conv::Conv
end

function UNet(in_channels::Int, out_channels::Int, embed_dim::Int)
    return UNet(
                GaussianFourierProjection(embed_dim),
                Chain(
                      Dense(embed_dim=>embed_dim, relu),
                      Dense(embed_dim=>embed_dim)
                     ),
                DownBlock(in_channels, 64, embed_dim),
                DownBlock(64, 128, embed_dim),
                Chain(
                      Conv((3,3), 128=>256, relu; pad=SamePad()),
                      Conv((3,3), 256=>256, relu; pad=SamePad())
                     ),
                UpBlock(256, 128, embed_dim),
                UpBlock(128, 64, embed_dim),
                Conv((1,1), 64=>out_channels)
               )
end

function (net::UNet)((x,t), ps, st)
    t_gf, st_gf = net.gauss_embed(t, ps.gauss_embed, st.gauss_embed)
    t_embed, st_mlp = net.time_mlp(t_gf, ps.time_mlp, st.time_mlp)

    (skip1, d1), st_d1 = net.down1((x,t_embed), ps.down1, st.down1)
    (skip2, d2), st_d2 = net.down2((d1,t_embed), ps.down2, st.down2)

    b, st_b = net.bottleneck(d2, ps.bottleneck, st.bottleneck)

    u2, st_u2 = net.up2((b, skip2, t_embed), ps.up2, st.up2)
    u1, st_u1 = net.up1((u2, skip1, t_embed), ps.up1, st.up1)

    out, st_f = net.final_conv(u1, ps.final_conv, st.final_conv)

    new_st = (
              gauss_embedding=st_gf,
              time_mlp=st_mlp,
              down1=st_d1, 
              down2=st_d2, 
              bottleneck=st_b, 
              up2=st_u2, 
              up1=st_u1, 
              final_conv=st_f
             )
    return out, new_st
end
