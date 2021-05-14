
struct PETScSolver <: LinearSolver
  ksp::Ref{KSP}
  comm::MPI.Comm
end

function PETScSolver(comm::MPI.Comm)
  ksp = _manage_mem(Ref{KSP}())
  @check_error_code PETSC.KSPCreate(comm,ksp)
  PETScSolver(ksp,comm)
end

PETScSolver() = PETScSolver(MPI.COMM_WORLD)

struct PETScSolverSS <: SymbolicSetup
  ksp::Ref{KSP}
  comm::MPI.Comm
end

struct PETScSolverNS <: NumericalSetup
  ksp::Ref{KSP}
  comm::MPI.Comm
  mat::Ref{Mat}
  rhs::Ref{Vec}
  sol::Ref{Vec}
end

function Algebra.symbolic_setup(solver::PETScSolver,mat::AbstractMatrix)
  PETScSolverSS(solver.ksp,solver.comm)
end

function Algebra.numerical_setup(
  ss::PETScSolverSS,A::SparseMatrixCSR{0,PetscScalar,PetscInt})

  ksp = ss.ksp
  comm = ss.comm
  mat = _manage_mem(Ref{Mat}())
  rhs = _manage_mem(Ref{Vec}())
  sol = _manage_mem(Ref{Vec}())
  bs = 1
  nrows, ncols = size(A)
  i = A.rowptr
  j = A.colval
  a = A.nzval
  @check_error_code PETSC.VecCreateSeqWithArray(comm,bs,nrows,C_NULL,rhs)
  @check_error_code PETSC.VecCreateSeqWithArray(comm,bs,ncols,C_NULL,sol)
  @check_error_code PETSC.MatCreateSeqAIJWithArrays(comm,nrows,ncols,i,j,a,mat)
  @check_error_code PETSC.KSPSetOperators(ksp[],mat[],mat[])
  @check_error_code PETSC.KSPSetUp(ksp[])
  PETScSolverNS(ksp,comm,mat,rhs,sol)
end

function Algebra.solve!(x::Vector{PetscScalar},ns::PETScSolverNS,b::Vector{PetscScalar})
  @check_error_code PETSC.VecPlaceArray(ns.rhs[],b)
  @check_error_code PETSC.VecPlaceArray(ns.sol[],x)
  @check_error_code PETSC.KSPSolve(ns.ksp[],ns.rhs[],ns.sol[])
  @check_error_code PETSC.VecResetArray(ns.rhs[])
  @check_error_code PETSC.VecResetArray(ns.sol[])
  x
end

function Algebra.numerical_setup!(ns::PETScSolverNS,A::SparseMatrixCSR{0,PetscScalar,PetscInt})
  nrows, ncols = size(A)
  i = A.rowptr
  j = A.colval
  a = A.nzval
  @check_error_code PETSC.MatDestroy(ns.mat)
  @check_error_code PETSC.MatCreateSeqAIJWithArrays(ns.comm,nrows,ncols,i,j,a,ns.mat)
  @check_error_code PETSC.KSPSetOperators(ns.ksp[],ns.mat[],ns.mat[])
  @check_error_code PETSC.KSPSetUp(ns.ksp[])
  ns
end

# with conversions

function Algebra.numerical_setup(ss::PETScSolverSS,A::AbstractMatrix)
  _A = convert(SparseMatrixCSR{0,PetscScalar,PetscInt},A)
  numerical_setup(ss,_A)
end

function Algebra.solve!(x::AbstractVector,ns::PETScSolverNS,b::AbstractVector)
  _x = convert(Vector{PetscScalar},x)
  _b = convert(Vector{PetscScalar},b)
  solve!(_x,ns,_b)
  if x !== _x
    x.=_x
  end
  x
end

function Algebra.numerical_setup!(ns::PETScSolverNS,A::AbstractMatrix)
  _A = convert(SparseMatrixCSR{0,PetscScalar,PetscInt},A)
  numerical_setup!(ns,_A)
end
