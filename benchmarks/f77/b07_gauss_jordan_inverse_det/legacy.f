C     ==================================================================
C     GJINV  -  MATRIX INVERSE AND DETERMINANT BY GAUSS-JORDAN
C     ELIMINATION WITH FULL PIVOTING
C
C     A IS REPLACED BY ITS INVERSE.  DET RECEIVES THE DETERMINANT OF
C     THE ORIGINAL MATRIX.  IERR = 0 NORMAL, 1 SINGULAR MATRIX (A IS
C     THEN DESTROYED AND DET = 0), 2 N OUTSIDE 1..NMAX.
C
C     THE PIVOT OF STEP I IS THE LARGEST ELEMENT IN THE ROWS AND
C     COLUMNS NOT USED SO FAR; WHEN SEVERAL ELEMENTS TIE THE LAST ONE
C     FOUND IS TAKEN.  ROW INTERCHANGES ARE DONE AT ONCE, COLUMN
C     INTERCHANGES ARE ONLY RECORDED IN INDXR/INDXC AND UNDONE IN
C     REVERSE ORDER AT THE END.
C     ==================================================================
      SUBROUTINE GJINV(A, LDA, N, DET, IERR)
      INTEGER LDA, N, IERR
      DOUBLE PRECISION A(LDA,N), DET
      INTEGER NMAX
      PARAMETER (NMAX = 50)
      INTEGER IPIV(NMAX), INDXR(NMAX), INDXC(NMAX)
      DOUBLE PRECISION BIG, DUM, PIVINV, T
      INTEGER I, ICOL, IROW, J, K, L, LL
C
      IERR = 0
      DET = 1.0D0
      IF (N .LT. 1 .OR. N .GT. NMAX) THEN
         IERR = 2
         RETURN
      END IF
      DO 10 J = 1, N
         IPIV(J) = 0
   10 CONTINUE
      IROW = 1
      ICOL = 1
      DO 90 I = 1, N
C        LARGEST ELEMENT AMONG THE UNUSED ROWS AND COLUMNS
         BIG = 0.0D0
         DO 30 J = 1, N
            IF (IPIV(J) .NE. 1) THEN
               DO 20 K = 1, N
                  IF (IPIV(K) .EQ. 0) THEN
                     IF (DABS(A(J,K)) .GE. BIG) THEN
                        BIG = DABS(A(J,K))
                        IROW = J
                        ICOL = K
                     END IF
                  END IF
   20          CONTINUE
            END IF
   30    CONTINUE
         IPIV(ICOL) = IPIV(ICOL) + 1
C        PUT THE PIVOT ON THE DIAGONAL BY SWAPPING ROWS
         IF (IROW .NE. ICOL) THEN
            DO 40 L = 1, N
               T = A(IROW,L)
               A(IROW,L) = A(ICOL,L)
               A(ICOL,L) = T
   40       CONTINUE
            DET = -DET
         END IF
         INDXR(I) = IROW
         INDXC(I) = ICOL
         IF (A(ICOL,ICOL) .EQ. 0.0D0) THEN
            IERR = 1
            DET = 0.0D0
            RETURN
         END IF
         DET = DET*A(ICOL,ICOL)
         PIVINV = 1.0D0/A(ICOL,ICOL)
         A(ICOL,ICOL) = 1.0D0
         DO 50 L = 1, N
            A(ICOL,L) = A(ICOL,L)*PIVINV
   50    CONTINUE
         DO 80 LL = 1, N
            IF (LL .NE. ICOL) THEN
               DUM = A(LL,ICOL)
               A(LL,ICOL) = 0.0D0
               DO 70 L = 1, N
                  A(LL,L) = A(LL,L) - A(ICOL,L)*DUM
   70          CONTINUE
            END IF
   80    CONTINUE
   90 CONTINUE
C     UNDO THE COLUMN INTERCHANGES, LAST PIVOT FIRST
      DO 120 L = N, 1, -1
         IF (INDXR(L) .NE. INDXC(L)) THEN
            DO 110 K = 1, N
               T = A(K,INDXR(L))
               A(K,INDXR(L)) = A(K,INDXC(L))
               A(K,INDXC(L)) = T
  110       CONTINUE
         END IF
  120 CONTINUE
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION MXRESI(A, AINV, LDA, N)
C     LARGEST ABSOLUTE ENTRY OF A*AINV - I.
      INTEGER LDA, N
      DOUBLE PRECISION A(LDA,N), AINV(LDA,N)
      DOUBLE PRECISION S
      INTEGER I, J, K
C
      MXRESI = 0.0D0
      DO 30 J = 1, N
         DO 20 I = 1, N
            S = 0.0D0
            DO 10 K = 1, N
               S = S + A(I,K)*AINV(K,J)
   10       CONTINUE
            IF (I .EQ. J) S = S - 1.0D0
            MXRESI = DMAX1(MXRESI, DABS(S))
   20    CONTINUE
   30 CONTINUE
      RETURN
      END
