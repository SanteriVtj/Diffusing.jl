@testset "UNet test" begin
    rng = Random.default_rng()
    in_channels = 1
    out_channels = 1
    batch_size = 4
    img_size = (32,32)
    t = .5f0

    x = randn(Float32, img_size..., in_channels, batch_size)

    model = @test_nowarn UNet(1,1,128)
    @test model isa UNet
    @test model isa Lux.AbstractLuxContainerLayer

    ps, st = @test_nowarn Lux.setup(rng, model)
    @test ps isa NamedTuple
    @test st isa NamedTuple
    @test !isempty(ps)
    @test !isempty(st)

    y_pred, new_st = @test_nowarn model((x, t), ps, st)

    @test size(y_pred) == size(x)
    @test eltype(y_pred) == Float32
    @test !any(isnan, y_pred)
end
