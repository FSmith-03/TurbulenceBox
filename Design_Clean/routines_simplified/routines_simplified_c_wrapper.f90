! This automatically generated Fortran wrapper file allows codes
! written in Fortran to be called directly from C and translates all
! C-style arguments into expected Fortran-style arguments (with
! assumed size, local type declarations, etc.).


SUBROUTINE C_MATRIX_ROTATE(THETAX, THETAY, THETAZ, R_DIM_1, R_DIM_2, R) BIND(C)
  USE ISO_FORTRAN_ENV, ONLY: INT64
  IMPLICIT NONE
  REAL, INTENT(IN) :: THETAX
  REAL, INTENT(IN) :: THETAY
  REAL, INTENT(IN) :: THETAZ
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: R_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: R_DIM_2
  REAL, ALLOCATABLE, SAVE, DIMENSION(:,:) :: R_LOCAL
  INTEGER(KIND=INT64), INTENT(OUT) :: R

  INTERFACE
    SUBROUTINE MATRIX_ROTATE(THETAX, THETAY, THETAZ, R)
      IMPLICIT NONE
      REAL, INTENT(IN) :: THETAX
      REAL, INTENT(IN) :: THETAY
      REAL, INTENT(IN) :: THETAZ
      REAL, INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: R
    END SUBROUTINE MATRIX_ROTATE
  END INTERFACE

  CALL MATRIX_ROTATE(THETAX, THETAY, THETAZ, R_LOCAL)
  
  R_DIM_1 = SIZE(R_LOCAL,1)
  R_DIM_2 = SIZE(R_LOCAL,2)
  R = LOC(R_LOCAL(1,1))
END SUBROUTINE C_MATRIX_ROTATE


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


SUBROUTINE C_TRIMMER_INDEX(POS_VECTORS_DIM_1, POS_VECTORS_DIM_2, POS_VECTORS, FIRST_AND_LAST_DIM_1, FIRST_AND_LAST) BIND(C)
  USE ISO_C_BINDING, ONLY: C_BOOL
  IMPLICIT NONE
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: POS_VECTORS_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: POS_VECTORS_DIM_2
  REAL, INTENT(IN), DIMENSION(POS_VECTORS_DIM_1,POS_VECTORS_DIM_2) :: POS_VECTORS
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: FIRST_AND_LAST_DIM_1
  INTEGER, INTENT(OUT), DIMENSION(FIRST_AND_LAST_DIM_1) :: FIRST_AND_LAST

  INTERFACE
    SUBROUTINE TRIMMER_INDEX(POS_VECTORS, FIRST_AND_LAST)
      IMPLICIT NONE
      REAL, INTENT(IN), DIMENSION(:,:) :: POS_VECTORS
      INTEGER, INTENT(OUT), DIMENSION(2) :: FIRST_AND_LAST
    END SUBROUTINE TRIMMER_INDEX
  END INTERFACE

  CALL TRIMMER_INDEX(POS_VECTORS, FIRST_AND_LAST)
END SUBROUTINE C_TRIMMER_INDEX


SUBROUTINE C_TRIMMER(FIRST_AND_LAST_DIM_1, FIRST_AND_LAST, XAXIS_DIM_1, XAXIS_DIM_2, XAXIS, XAXIS_TRIMMED_DIM_1, XAXIS_TRIMMED_DIM_&
&2, XAXIS_TRIMMED) BIND(C)
  USE ISO_FORTRAN_ENV, ONLY: INT64
  IMPLICIT NONE
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: FIRST_AND_LAST_DIM_1
  INTEGER, INTENT(IN), DIMENSION(FIRST_AND_LAST_DIM_1) :: FIRST_AND_LAST
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: XAXIS_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: XAXIS_DIM_2
  REAL, ALLOCATABLE, SAVE, DIMENSION(:,:) :: XAXIS_LOCAL
  INTEGER(KIND=INT64), INTENT(OUT) :: XAXIS
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: XAXIS_TRIMMED_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: XAXIS_TRIMMED_DIM_2
  REAL, ALLOCATABLE, SAVE, DIMENSION(:,:) :: XAXIS_TRIMMED_LOCAL
  INTEGER(KIND=INT64), INTENT(OUT) :: XAXIS_TRIMMED

  INTERFACE
    SUBROUTINE TRIMMER(FIRST_AND_LAST, XAXIS, XAXIS_TRIMMED)
      IMPLICIT NONE
      INTEGER, INTENT(IN), DIMENSION(:) :: FIRST_AND_LAST
      REAL, INTENT(IN), ALLOCATABLE, DIMENSION(:,:) :: XAXIS
      REAL, INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: XAXIS_TRIMMED
    END SUBROUTINE TRIMMER
  END INTERFACE

  CALL TRIMMER(FIRST_AND_LAST, XAXIS_LOCAL, XAXIS_TRIMMED_LOCAL)
  
  XAXIS_TRIMMED_DIM_1 = SIZE(XAXIS_TRIMMED_LOCAL,1)
  XAXIS_TRIMMED_DIM_2 = SIZE(XAXIS_TRIMMED_LOCAL,2)
  XAXIS_TRIMMED = LOC(XAXIS_TRIMMED_LOCAL(1,1))
END SUBROUTINE C_TRIMMER


SUBROUTINE C_VELOCITY_CALC(XAXIS_TRIMMED_DIM_1, XAXIS_TRIMMED_DIM_2, XAXIS_TRIMMED, U_DIM_1, U, V_DIM_1, V, W_DIM_1, W) BIND(C)
  USE ISO_FORTRAN_ENV, ONLY: INT64
  IMPLICIT NONE
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: XAXIS_TRIMMED_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: XAXIS_TRIMMED_DIM_2
  REAL, INTENT(IN), DIMENSION(XAXIS_TRIMMED_DIM_1,XAXIS_TRIMMED_DIM_2) :: XAXIS_TRIMMED
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: U_DIM_1
  REAL, ALLOCATABLE, SAVE, DIMENSION(:) :: U_LOCAL
  INTEGER(KIND=INT64), INTENT(OUT) :: U
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: V_DIM_1
  REAL, ALLOCATABLE, SAVE, DIMENSION(:) :: V_LOCAL
  INTEGER(KIND=INT64), INTENT(OUT) :: V
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: W_DIM_1
  REAL, ALLOCATABLE, SAVE, DIMENSION(:) :: W_LOCAL
  INTEGER(KIND=INT64), INTENT(OUT) :: W

  INTERFACE
    SUBROUTINE VELOCITY_CALC(XAXIS_TRIMMED, U, V, W)
      IMPLICIT NONE
      REAL, INTENT(IN), DIMENSION(:,:) :: XAXIS_TRIMMED
      REAL, INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: U
      REAL, INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: V
      REAL, INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: W
    END SUBROUTINE VELOCITY_CALC
  END INTERFACE

  CALL VELOCITY_CALC(XAXIS_TRIMMED, U_LOCAL, V_LOCAL, W_LOCAL)
  
  U_DIM_1 = SIZE(U_LOCAL,1)
  U = LOC(U_LOCAL(1))
  
  V_DIM_1 = SIZE(V_LOCAL,1)
  V = LOC(V_LOCAL(1))
  
  W_DIM_1 = SIZE(W_LOCAL,1)
  W = LOC(W_LOCAL(1))
END SUBROUTINE C_VELOCITY_CALC


SUBROUTINE C_VECTOR_SUMS(U_DIM_1, U, V_DIM_1, V, W_DIM_1, W, U_TOTAL_DIM_1, U_TOTAL, V_TOTAL_DIM_1, V_TOTAL, W_TOTAL_DIM_1, W_TOTAL&
&, FIRST_INDEX, LAST_INDEX) BIND(C)
  IMPLICIT NONE
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: U_DIM_1
  REAL, INTENT(IN), DIMENSION(U_DIM_1) :: U
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: V_DIM_1
  REAL, INTENT(IN), DIMENSION(V_DIM_1) :: V
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: W_DIM_1
  REAL, INTENT(IN), DIMENSION(W_DIM_1) :: W
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: U_TOTAL_DIM_1
  REAL, INTENT(OUT), DIMENSION(U_TOTAL_DIM_1) :: U_TOTAL
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: V_TOTAL_DIM_1
  REAL, INTENT(OUT), DIMENSION(V_TOTAL_DIM_1) :: V_TOTAL
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: W_TOTAL_DIM_1
  REAL, INTENT(OUT), DIMENSION(W_TOTAL_DIM_1) :: W_TOTAL
  INTEGER, INTENT(IN) :: FIRST_INDEX
  INTEGER, INTENT(IN) :: LAST_INDEX

  INTERFACE
    SUBROUTINE VECTOR_SUMS(U, V, W, U_TOTAL, V_TOTAL, W_TOTAL, FIRST_INDEX, LAST_INDEX)
      IMPLICIT NONE
      REAL, INTENT(IN), DIMENSION(:) :: U
      REAL, INTENT(IN), DIMENSION(:) :: V
      REAL, INTENT(IN), DIMENSION(:) :: W
      REAL, INTENT(OUT), DIMENSION(:) :: U_TOTAL
      REAL, INTENT(OUT), DIMENSION(:) :: V_TOTAL
      REAL, INTENT(OUT), DIMENSION(:) :: W_TOTAL
      INTEGER, INTENT(IN) :: FIRST_INDEX
      INTEGER, INTENT(IN) :: LAST_INDEX
    END SUBROUTINE VECTOR_SUMS
  END INTERFACE

  CALL VECTOR_SUMS(U, V, W, U_TOTAL, V_TOTAL, W_TOTAL, FIRST_INDEX, LAST_INDEX)
END SUBROUTINE C_VECTOR_SUMS


SUBROUTINE C_MAIN_CALCULATION(INPUT_INTS_DIM_1, INPUT_INTS, A_LIST_DIM_1, A_LIST_DIM_2, A_LIST, THETA_LIST_DIM_1, THETA_LIST_DIM_2,&
& THETA_LIST, VELOCITY_TOTAL_DIM_1, VELOCITY_TOTAL_DIM_2, VELOCITY_TOTAL) BIND(C)
  USE ISO_FORTRAN_ENV, ONLY: INT64
  IMPLICIT NONE
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: INPUT_INTS_DIM_1
  INTEGER, INTENT(IN), DIMENSION(INPUT_INTS_DIM_1) :: INPUT_INTS
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: A_LIST_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: A_LIST_DIM_2
  REAL, INTENT(INOUT), DIMENSION(A_LIST_DIM_1,A_LIST_DIM_2) :: A_LIST
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: THETA_LIST_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: THETA_LIST_DIM_2
  REAL, INTENT(INOUT), DIMENSION(THETA_LIST_DIM_1,THETA_LIST_DIM_2) :: THETA_LIST
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: VELOCITY_TOTAL_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(OUT) :: VELOCITY_TOTAL_DIM_2
  REAL, ALLOCATABLE, SAVE, DIMENSION(:,:) :: VELOCITY_TOTAL_LOCAL
  INTEGER(KIND=INT64), INTENT(OUT) :: VELOCITY_TOTAL

  INTERFACE
    SUBROUTINE MAIN_CALCULATION(INPUT_INTS, A_LIST, THETA_LIST, VELOCITY_TOTAL)
      IMPLICIT NONE
      INTEGER, INTENT(IN), DIMENSION(:) :: INPUT_INTS
      REAL, INTENT(INOUT), DIMENSION(:,:) :: A_LIST
      REAL, INTENT(INOUT), DIMENSION(:,:) :: THETA_LIST
      REAL, INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: VELOCITY_TOTAL
    END SUBROUTINE MAIN_CALCULATION
  END INTERFACE

  CALL MAIN_CALCULATION(INPUT_INTS, A_LIST, THETA_LIST, VELOCITY_TOTAL_LOCAL)
  
  VELOCITY_TOTAL_DIM_1 = SIZE(VELOCITY_TOTAL_LOCAL,1)
  VELOCITY_TOTAL_DIM_2 = SIZE(VELOCITY_TOTAL_LOCAL,2)
  VELOCITY_TOTAL = LOC(VELOCITY_TOTAL_LOCAL(1,1))
END SUBROUTINE C_MAIN_CALCULATION


SUBROUTINE C_SENSOR_LINE_GENERATOR(X_BOUNDARY, NXF, POS_VECTOR_DIM_1, POS_VECTOR_DIM_2, POS_VECTOR) BIND(C)
  USE ISO_C_BINDING, ONLY: C_BOOL
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: X_BOUNDARY
  INTEGER, INTENT(IN) :: NXF
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: POS_VECTOR_DIM_1
  INTEGER(KIND=SELECTED_INT_KIND(18)), INTENT(IN) :: POS_VECTOR_DIM_2
  REAL, DIMENSION(POS_VECTOR_DIM_1,POS_VECTOR_DIM_2) :: POS_VECTOR

  INTERFACE
    FUNCTION SENSOR_LINE_GENERATOR(X_BOUNDARY, NXF) RESULT(POS_VECTOR)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: X_BOUNDARY
      INTEGER, INTENT(IN) :: NXF
      REAL, DIMENSION(3,NXF) :: POS_VECTOR
    END FUNCTION SENSOR_LINE_GENERATOR
  END INTERFACE

  POS_VECTOR = SENSOR_LINE_GENERATOR(X_BOUNDARY, NXF)
END SUBROUTINE C_SENSOR_LINE_GENERATOR

