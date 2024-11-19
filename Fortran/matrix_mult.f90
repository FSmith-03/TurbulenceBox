! This module provides various testbed routines for demonstrating
! the simplicity of Fortran code usage with `fmodpy`.
! 
! Contains:
! 
!   MATRIX_MULTIPLY  --  A routine for multiplying two matrices of floats.
! 

SUBROUTINE MATRIX_MULTIPLY(A,B,OUT)
    ! This subroutine multiplies the matrices A and B.
    ! 
    ! INPUT:
    !   A(M,N)  --  A 2D matrix of 64 bit floats.
    !   B(N,P)  --  A 2D matrix of 64 bit floats,
    ! 
    ! OUTPUT:
    !   OUT(M,P)  --  The matrix that is the result of (AB).
    ! 
    USE ISO_FORTRAN_ENV, ONLY: REAL64 ! <- Get a float64 type.
    IMPLICIT NONE  ! <- Make undefined variable usage raise errors.
    REAL(KIND=REAL64), INTENT(IN),  DIMENSION(:,:) :: A, B
    REAL(KIND=REAL64), INTENT(OUT), DIMENSION(SIZE(A,1),SIZE(B,2)) :: OUT
  
    ! Compute the matrix multiplication of A and B.
    OUT(:,:) = MATMUL(A,B)
  
  END SUBROUTINE MATRIX_MULTIPLY