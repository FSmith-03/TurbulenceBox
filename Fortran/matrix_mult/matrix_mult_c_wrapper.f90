! This automatically generated Fortran wrapper file allows codes
! written in Fortran to be called directly from C and translates all
! C-style arguments into expected Fortran-style arguments (with
! assumed size, local type declarations, etc.).


SUBROUTINE C_MATRIX_MULTIPLY(A_DIM_1, A_DIM_2, A, B_DIM_1, B_DIM_2, B, OUT_DIM_1, OUT_DIM_2, OUT) BIND(C)
  USE ISO_FORTRAN_ENV , ONLY : REAL64
  USE ISO_C_BINDING, ONLY: C_BOOL
  IMPLICIT NONE
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: A_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: A_DIM_2
  REAL(KIND=REAL64), INTENT(IN), DIMENSION(A_DIM_1,A_DIM_2) :: A
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: B_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: B_DIM_2
  REAL(KIND=REAL64), INTENT(IN), DIMENSION(B_DIM_1,B_DIM_2) :: B
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: OUT_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: OUT_DIM_2
  REAL(KIND=REAL64), INTENT(OUT), DIMENSION(OUT_DIM_1,OUT_DIM_2) :: OUT

  INTERFACE
    SUBROUTINE MATRIX_MULTIPLY(A, B, OUT)
      ! This subroutine multiplies the matrices A and B.
      !
      ! INPUT:
      !   A(M,N)  --  A 2D matrix of 64 bit floats.
      !   B(N,P)  --  A 2D matrix of 64 bit floats,
      !
      ! OUTPUT:
      !   OUT(M,P)  --  The matrix that is the result of (AB).
      USE ISO_FORTRAN_ENV , ONLY : REAL64
      IMPLICIT NONE
      REAL(KIND=REAL64), INTENT(IN), DIMENSION(:,:) :: A
      REAL(KIND=REAL64), INTENT(IN), DIMENSION(:,:) :: B
      REAL(KIND=REAL64), INTENT(OUT), DIMENSION(SIZE(A,1),SIZE(B,2)) :: OUT
    END SUBROUTINE MATRIX_MULTIPLY
  END INTERFACE

  CALL MATRIX_MULTIPLY(A, B, OUT)
END SUBROUTINE C_MATRIX_MULTIPLY

