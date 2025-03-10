using SparseMatricesCSR
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapPETSc
using GridapPETSc: PETSC
using PartitionedArrays
using Test
using SparseMatricesCSR


function main(distribute,nparts)
  parts = distribute(LinearIndices((prod(nparts),)))
  
  options = "-pc_type gamg -ksp_type gmres -ksp_atol 0.0 -ksp_rtol 1e-14 -ksp_max_it 1000"
  GridapPETSc.with(args=split(options)) do
      
      # Define the parameters to run the problem
      L = 1 # Length of the cavity  
      h = 1 # Height of the cavity
      f = 8e3 # Frequency
      c = 3000 # Speed of sound in the medium
      k = 2*π*f/c # Wavenumber
      θ = π/4 # Angle of the wave
      k_x = k*cos(θ)
      k_y = k*sin(θ)
      λ = 1/k # Wavelength
      N = ceil(15*h/λ) # Number of elements per wavelength

      # Define the analytical solution and the Laplacian of the analytical solution to contruct the source term of the manufactured solution
      u(x) = exp(1im*(k_x*x[1] + k_y*x[2])) * (L*x[1] - x[1]^2) * (h*x[2] - x[2]^2)
      lap_u(x) = (h*x[2]-x[2]^2) * (k_x^2*x[1]^2 - (4im*k_x + k_x^2*L)*x[1] + (2im*k_x*L - 2)) * exp(1im*(k_x*x[1] + k_y*x[2])) + 
                 (L*x[1]-x[1]^2) * (k_y^2*x[2]^2 - (4im*k_y + k_y^2*h)*x[2] + (2im*k_y*h - 2)) * exp(1im*(k_x*x[1] + k_y*x[2]))

      domain = (0,L,0,h)
      partition = (N,N)
      model = CartesianDiscreteModel(parts, nparts, domain, partition)

      labels = get_face_labeling(model)
      add_tag_from_tags!(labels,"sides", [5,6,7,8])

      # Define the finite element space: Lagrange of order 1
      order = 1
      reffe = ReferenceFE(lagrangian, Float64, order)
      V = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["sides"], vector_type=Vector{ComplexF64})

      # Define the trial function with Dirichlet boundary conditions
      U = TrialFESpace(V, [u])

      # Define the triangulation
      Ω = Triangulation(model)

      # Define the measure for the fluid and porous domains
      degree = 2
      dΩ = Measure(Ω, degree)

      f(x) = -lap_u(x) - k^2*u(x)

      a(u,v) = ∫( (1.0+0.0im)*(∇(v) ⊙ ∇(u)) )*dΩ - ∫( k^2*(1.0+0.0im)*(u*v) )*dΩ
      b(v) = ∫( v*f )*dΩ

      op = AffineFEOperator(a, b, U, V)

      ls = PETScLinearSolver()
      fesolver = FESolver(ls)
      uh = zero(U)
      uh, _ = solve!(uh, fesolver, op)
      u_ex = CellField(u, Ω)
      eh = u - uh
      @test 100 * sqrt(sum(∫( abs2(eh) )dΩ))/sqrt(sum(∫( abs2(u_ex) )dΩ)) < 1
  end
 end
