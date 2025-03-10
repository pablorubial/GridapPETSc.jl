module GridapPETScSequentialTests

using Test
using MPI
using GridapPETSc

@time @testset "PETSC" begin include("PETSCTests.jl") end

@time @testset "PETScArrays" begin include("PETScArraysTests.jl") end

@time @testset "PartitionedArrays (sequential)" begin include("PartitionedArraysTests.jl") end

@time @testset "PETScLinearSolvers" begin include("PETScLinearSolversTests.jl") end

@time @testset "PETScNonLinearSolvers" begin include("PETScNonlinearSolversTests.jl") end

@time @testset "PETScAssembly" begin include("PETScAssemblyTests.jl") end

@time @testset "PoissonDriver" begin include("PoissonDriver.jl") end

@time @testset "ElasticityDriver" begin include("ElasticityDriver.jl") end

@time @testset "DarcyDriver" begin include("DarcyDriver.jl") end

@time @testset "PLaplacianDriver" begin include("PLaplacianDriver.jl") end

@time @testset "HelmholtzPMLDriver" begin PetscScalar == ComplexF64 ? include("HelmholtzDriver.jl") : nothing end

# Partitioned in sequential mode

@time @testset "PoissonTests" begin include("PoissonTests.jl") end

@time @testset "PLaplacianTests" begin include("PLaplacianTests.jl") end

if MPI.Initialized()
    MPI.Finalize()
end

end # module


