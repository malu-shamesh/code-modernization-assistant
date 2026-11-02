C**********************************************************************
C     RUNNING SUMS AND MERGING OF SORTED VECTORS
C
C     CUMSUM  S(I) = X(1) + ... + X(I); S MAY BE THE SAME ARRAY AS X
C     IFIRST  FIRST INDEX AT WHICH THE RUNNING SUM REACHES A LIMIT
C     CUMCOL  RUNNING SUMS DOWN EVERY COLUMN OF A MATRIX HELD WITH
C             LEADING DIMENSION LDA
C     RCOPY   COPIES N ELEMENTS
C     MRGSRT  MERGES TWO ASCENDING VECTORS INTO ONE; ON EQUAL KEYS THE
C             ELEMENT OF A COMES FIRST
C**********************************************************************
      SUBROUTINE CUMSUM(N, X, S)
      INTEGER N
      REAL X(N), S(N)
      REAL ACC
      INTEGER I
C
      ACC = 0.0
      DO 10 I = 1, N
         ACC = ACC + X(I)
         S(I) = ACC
   10 CONTINUE
      RETURN
      END
C
C     IFIRST - 0 WHEN THE LIMIT IS NEVER REACHED
C
      INTEGER FUNCTION IFIRST(N, X, THRESH)
      INTEGER N
      REAL X(N), THRESH
      REAL ACC
      INTEGER I
C
      IFIRST = 0
      ACC = 0.0
      DO 10 I = 1, N
         ACC = ACC + X(I)
         IF (ACC .GE. THRESH) THEN
            IFIRST = I
            RETURN
         END IF
   10 CONTINUE
      RETURN
      END
C
C     CUMCOL - A(I,J) BECOMES A(1,J) + ... + A(I,J) FOR I = 1..M AND
C     J = 1..N.  BOTH LOOPS END AT THE SAME LABEL.  ROWS M+1..LDA ARE
C     NOT TOUCHED.
C
      SUBROUTINE CUMCOL(LDA, M, N, A)
      INTEGER LDA, M, N
      REAL A(LDA, N)
      INTEGER I, J
C
      DO 10 J = 1, N
      DO 10 I = 2, M
   10 A(I, J) = A(I, J) + A(I-1, J)
      RETURN
      END
C
      SUBROUTINE RCOPY(N, A, B)
      REAL A(*), B(*)
      DO 10 I = 1, N
         B(I) = A(I)
   10 CONTINUE
      RETURN
      END
C
C     MRGSRT - C MUST HOLD NA+NB ELEMENTS.  WHEN ONE INPUT RUNS OUT THE
C     REST OF THE OTHER ONE IS COPIED WITH THE SUB-ARRAY A(I) OR B(J)
C     PASSED AS THE START OF THE SOURCE AND C(K) AS THE TARGET.
C
      SUBROUTINE MRGSRT(NA, A, NB, B, C, NC)
      INTEGER NA, NB, NC
      REAL A(NA), B(NB), C(*)
      INTEGER I, J, K
C
      I = 1
      J = 1
      K = 1
   10 IF (I .GT. NA) GO TO 30
      IF (J .GT. NB) GO TO 20
      IF (B(J) .LT. A(I)) THEN
         C(K) = B(J)
         J = J + 1
      ELSE
         C(K) = A(I)
         I = I + 1
      END IF
      K = K + 1
      GO TO 10
   20 CALL RCOPY(NA - I + 1, A(I), C(K))
      GO TO 40
   30 CALL RCOPY(NB - J + 1, B(J), C(K))
   40 NC = NA + NB
      RETURN
      END
