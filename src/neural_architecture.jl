# Down and upsampling blocks

struct DownBlock <: AbstractLuxContainerLayer{(:conv,)}
    conv::Chain
end

function DownBlock(in_chs::Int, out_chs::Int)
    conv = Chain(
                 Conv((3,3), in_chs=>out_chs, relu; pad=SamePad()),
                 Conv((3,3), out_chs=>out_chs, relu; pad=SamePad())
                )
    return DownBlock(conv)
end

function (d::DownBlock)(x, ps, st)
    features, st_conv = d.conv(x, ps.conv, st.conv)
    downsampled = MaxPool((2,2))(features)
    return (features, downsampled), (conv = st_conv,)
end

struct UpBlock <: Lux.AbstractLuxContainerLayer{(:upsample, :conv)}
    upsample::Lux.Upsample{nothing, :bilinear}
    conv::Chain
end

function UpBlock(in_chs::Int, out_chs::Int)
    upsample = Lux.Upsample(2; mode=:bilinear, align_corners=false)
    conv = Chain(
                 Conv((3,3), (in_chs + out_chs)=>out_chs, relu; pad=SamePad()),
                 Conv((3,3), out_chs=>out_chs, relu; pad=SamePad())
                )
    return UpBlock(upsample, conv)
end

function (u::UpBlock)((x, skip), ps, st)
    x_up, st_up = u.upsample(x, ps.upsample, st.upsample)
    x_concat = cat(x_up, skip; dims=3)
    out, st_conv = u.conv(x_concat, ps.conv, st.conv)
    return out, (upsample=st_up, conv=st_conv)
end

# UNet definition

struct UNet <: Lux.AbstractLuxContainerLayer{(:down1, :down2, :bottleneck, :up2, :up1, :final_conv)}
    down1::DownBlock
    down2::DownBlock
    bottleneck::Chain
    up2::UpBlock
    up1::UpBlock
    final_conv::Conv
end

function Unet(in_channels::Int, out_channels::Int)
    return UNet(
                DownBlock(in_channels, 64),
                DownBlock(64, 128),
                Chain(
                      Conv((3,3), 128=>256, relu; pad=SamePad()),
                      Conv((3,3), 256=>256, relu; pad=SamePad())
                     ),
                UpBlock(256, 128),
                UpBlock(128, 64),
                Conv((1,1), 64=>out_channels)
               )
end

function (net::UNet)(x, ps, st)
    (skip1, d1), st_d1 = net.down1(x, ps.down1, st.down1)
    (skip2, d2), st_d2 = net.down2(x, ps.down2, st.down2)

    b, st_b = net.bottleneck(d2, ps.bottleneck, st.bottleneck)

    u2, st_u2 = net.up2((b, skip2), ps.up2, st.up2)
    u1, st_u1 = net.up1((u2, skip1), ps.up1, st.up1)

    out, st_f = net.final_conv(u1, ps.final_conv, st.final_conv)

    new_st = (down1=st_d1, down2=sd_d2, bottleneck=st_b, up2=st_u2, up1=st_u1, final_conv=st_f)
    return out, new_st
end
