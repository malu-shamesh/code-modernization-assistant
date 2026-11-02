C     ==================================================================
C     CHOLPK  -  CHOLESKY FACTORISATION A = L*L' OF A SYMMETRIC
C     POSITIVE DEFINITE MATRIX, AND THE TRIANGULAR SOLVES
C
C     CHOLDC  OVERWRITES THE LOWER TRIANGLE OF A WITH L.  THE STRICTLY
C             UPPER TRIANGLE IS NEVER REFERENCED.  DMIN RECEIVES THE
C             SMALLEST DIAGONAL ELEMENT OF L.  IF A IS NOT POSITIVE
C             DEFINITE THE ROUTINE TAKES ALTERNATE RETURN 1 AND DMIN
C             HOLDS THE OFFENDING PIVOT (ZERO OR NEGATIVE).
C     CHOLSL  SOLVES L*L'*X = B IN PLACE, GIVEN THE FACTOR.
C     SPDSOL  FACTORS AND SOLVES, PASSING THE ALTERNATE RETURN ON.
C     CHOLDT  NATURAL LOGARITHM OF DET(A) FROM THE FACTOR.
C     ==================================================================
      SUBROUTINE CHOLDC(A, LDA, N, DMIN, *)
      INTEGER LDA, N
      DOUBLE PRECISION A(LDA,N), DMIN
      DOUBLE PRECISION S
      INTEGER I, J, K
C
      DMIN = 0.0D0
      DO 30 J = 1, N
C        DIAGONAL ELEMENT
         S = A(J,J)
         DO 10 K = 1, J-1
            S = S - A(J,K)*A(J,K)
   10    CONTINUE
         IF (S .LE. 0.0D0) THEN
            DMIN = S
            RETURN 1
         END IF
         A(J,J) = DSQRT(S)
         IF (J .EQ. 1 .OR. A(J,J) .LT. DMIN) DMIN = A(J,J)
C        COLUMN BELOW THE DIAGONAL
         DO 25 I = J+1, N
            S = A(I,J)
            DO 20 K = 1, J-1
               S = S - A(I,K)*A(J,K)
   20       CONTINUE
            A(I,J) = S/A(J,J)
   25    CONTINUE
   30 CONTINUE
      RETURN
      END
C
      SUBROUTINE CHOLSL(A, LDA, N, B)
      INTEGER LDA, N
      DOUBLE PRECISION A(LDA,N), B(N)
      DOUBLE PRECISION S
      INTEGER I, K
C
C     FORWARD SUBSTITUTION: L*Y = B
      DO 20 I = 1, N
         S = B(I)
         DO 10 K = 1, I-1
            S = S - A(I,K)*B(K)
   10    CONTINUE
         B(I) = S/A(I,I)
   20 CONTINUE
C     BACK SUBSTITUTION: L'*X = Y
      DO 50 I = N, 1, -1
         S = B(I)
         DO 40 K = I+1, N
            S = S - A(K,I)*B(K)
   40    CONTINUE
         B(I) = S/A(I,I)
   50 CONTINUE
      RETURN
      END
C
      SUBROUTINE SPDSOL(A, LDA, N, B, DMIN, *)
      INTEGER LDA, N
      DOUBLE PRECISION A(LDA,N), B(N), DMIN
C
      CALL CHOLDC(A, LDA, N, DMIN, *90)
      CALL CHOLSL(A, LDA, N, B)
      RETURN
   90 RETURN 1
      END
C
      DOUBLE PRECISION FUNCTION CHOLDT(A, LDA, N)
      INTEGER LDA, N
      DOUBLE PRECISION A(LDA,N)
      INTEGER I
C
      CHOLDT = 0.0D0
      DO 10 I = 1, N
         CHOLDT = CHOLDT + 2.0D0*DLOG(A(I,I))
   10 CONTINUE
      RETURN
      END
