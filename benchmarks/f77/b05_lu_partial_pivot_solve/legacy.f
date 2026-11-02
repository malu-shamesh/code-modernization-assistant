C     ==================================================================
C     LUPACK  -  LU FACTORISATION WITH PARTIAL PIVOTING (LINPACK STYLE)
C
C     LUFAC   FACTORS A = P*L*U IN PLACE.  THE MULTIPLIERS ARE STORED
C             BELOW THE DIAGONAL WITH THEIR SIGN CHANGED (-L(I,K)).
C             IPVT(K) IS THE ROW THAT WAS SWAPPED WITH ROW K AT STEP K.
C             INFO = 0, OTHERWISE THE INDEX OF THE LAST ZERO PIVOT.
C     LUSOL   SOLVES A*X = B FROM THE FACTORS; B IS OVERWRITTEN BY X.
C     LUDET   DETERMINANT FROM THE FACTORS.
C     LUSLV   FACTORS AND SOLVES IN ONE CALL.
C     ==================================================================
      SUBROUTINE LUFAC(A, LDA, N, IPVT, INFO)
      INTEGER LDA, N, IPVT(*), INFO
      DOUBLE PRECISION A(LDA,*)
      DOUBLE PRECISION T
      INTEGER I, J, K, L, NM1
C
      INFO = 0
      IF (N .LT. 1) RETURN
      NM1 = N - 1
      IF (NM1 .LT. 1) GO TO 70
      DO 60 K = 1, NM1
C        FIND THE PIVOT ROW L
         L = K
         DO 10 I = K+1, N
            IF (DABS(A(I,K)) .GT. DABS(A(L,K))) L = I
   10    CONTINUE
         IPVT(K) = L
         IF (A(L,K) .EQ. 0.0D0) GO TO 50
C        INTERCHANGE ROWS WITHIN COLUMN K
         IF (L .EQ. K) GO TO 20
            T = A(L,K)
            A(L,K) = A(K,K)
            A(K,K) = T
   20    CONTINUE
C        NEGATED MULTIPLIERS
         T = -1.0D0/A(K,K)
         DO 30 I = K+1, N
            A(I,K) = T*A(I,K)
   30    CONTINUE
C        ROW ELIMINATION, COLUMN BY COLUMN
         DO 45 J = K+1, N
            T = A(L,J)
            IF (L .EQ. K) GO TO 35
               A(L,J) = A(K,J)
               A(K,J) = T
   35       CONTINUE
            DO 40 I = K+1, N
               A(I,J) = A(I,J) + T*A(I,K)
   40       CONTINUE
   45    CONTINUE
         GO TO 60
   50    CONTINUE
         INFO = K
   60 CONTINUE
   70 CONTINUE
      IPVT(N) = N
      IF (A(N,N) .EQ. 0.0D0) INFO = N
      RETURN
      END
C
      SUBROUTINE LUSOL(A, LDA, N, IPVT, B)
      INTEGER LDA, N, IPVT(*)
      DOUBLE PRECISION A(LDA,*), B(*)
      DOUBLE PRECISION T
      INTEGER I, K, KB, L, NM1
C
      NM1 = N - 1
      IF (NM1 .LT. 1) GO TO 30
C     FORWARD ELIMINATION: SOLVE L*Y = P*B
      DO 20 K = 1, NM1
         L = IPVT(K)
         T = B(L)
         IF (L .EQ. K) GO TO 10
            B(L) = B(K)
            B(K) = T
   10    CONTINUE
         DO 15 I = K+1, N
            B(I) = B(I) + T*A(I,K)
   15    CONTINUE
   20 CONTINUE
   30 CONTINUE
C     BACK SUBSTITUTION: SOLVE U*X = Y
      DO 50 KB = 1, N
         K = N + 1 - KB
         B(K) = B(K)/A(K,K)
         T = -B(K)
         DO 40 I = 1, K-1
            B(I) = B(I) + T*A(I,K)
   40    CONTINUE
   50 CONTINUE
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION LUDET(A, LDA, N, IPVT)
      INTEGER LDA, N, IPVT(*)
      DOUBLE PRECISION A(LDA,*)
      INTEGER K
C
      LUDET = 1.0D0
      DO 10 K = 1, N
         IF (IPVT(K) .NE. K) LUDET = -LUDET
         LUDET = LUDET*A(K,K)
   10 CONTINUE
      RETURN
      END
C
      SUBROUTINE LUSLV(A, LDA, N, B, IPVT, INFO)
      INTEGER LDA, N, IPVT(*), INFO
      DOUBLE PRECISION A(LDA,*), B(*)
C
      CALL LUFAC(A, LDA, N, IPVT, INFO)
      IF (INFO .EQ. 0) CALL LUSOL(A, LDA, N, IPVT, B)
      RETURN
      END
