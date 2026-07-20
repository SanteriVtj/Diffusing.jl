@testset "Diffusion processes" begin
    x = randn(10)
    t = rand()

    vpsde = VariancePreservingSDE(1e-4, 1, cos)
    @test size(drift(vpsde, x, t)) == size(x)
    @test size(diffusion(vpsde, x, t)) == ()

    vesde = VarianceExplodingSDE(1e-4, 1)
    @test size(drift(vesde, x, t)) == size(x)
    @test all(drift(vesde, x, t) .== 0)
    @test size(diffusion(vesde, x, t)) == ()
end
