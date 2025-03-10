module HelmholtzTestsRun
 include("mpiexec.jl")
 run_mpi_driver(procs=4,file="HelmholtzTests.jl")
end # module
