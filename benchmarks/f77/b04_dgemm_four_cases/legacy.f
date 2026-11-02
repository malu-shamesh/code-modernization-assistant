C     ==================================================================
C     DGEMM  -  C := ALPHA*OP(A)*OP(B) + BETA*C
C
C     OP(X) IS X OR X-TRANSPOSE.  OP(A) IS M BY K, OP(B) IS K BY N,
C     C IS M BY N.  ALL THREE MATRICES ARE STORED BY COLUMNS WITH THEIR
C     OWN LEADING DIMENSIONS, SO A TRANSPOSED OPERAND IS STORED WITH
C     SWAPPED EXTENTS.  INFO = 0 ON SUCCESS, ELSE THE POSITION OF THE
C     FIRST ILLEGAL ARGUMENT.
C     ==================================================================
      SUBROUTINE DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B,
     +                 LDB, BETA, C, LDC, INFO)
      CHARACTER*1 TRANSA, TRANSB
      INTEGER M, N, K, LDA, LDB, LDC, INFO
      DOUBLE PRECISION ALPHA, BETA
      DOUBLE PRECISION A(LDA,*), B(LDB,*), C(LDC,*)
      DOUBLE PRECISION ONE, ZERO
      PARAMETER (ONE = 1.0D0, ZERO = 0.0D0)
      DOUBLE PRECISION TEMP
      INTEGER I, J, L, NROWA, NROWB
      LOGICAL NOTA, NOTB, LSAME
      EXTERNAL LSAME
C
      NOTA = LSAME(TRANSA,'N')
      NOTB = LSAME(TRANSB,'N')
      NROWA = K
      IF (NOTA) NROWA = M
      NROWB = N
      IF (NOTB) NROWB = K
      INFO = 0
      IF (.NOT.NOTA .AND. .NOT.LSAME(TRANSA,'C') .AND.
     +    .NOT.LSAME(TRANSA,'T')) THEN
         INFO = 1
      ELSE IF (.NOT.NOTB .AND. .NOT.LSAME(TRANSB,'C') .AND.
     +         .NOT.LSAME(TRANSB,'T')) THEN
         INFO = 2
      ELSE IF (M .LT. 0) THEN
         INFO = 3
      ELSE IF (N .LT. 0) THEN
         INFO = 4
      ELSE IF (K .LT. 0) THEN
         INFO = 5
      ELSE IF (LDA .LT. MAX(1,NROWA)) THEN
         INFO = 8
      ELSE IF (LDB .LT. MAX(1,NROWB)) THEN
         INFO = 10
      ELSE IF (LDC .LT. MAX(1,M)) THEN
         INFO = 13
      END IF
      IF (INFO .NE. 0) RETURN
      IF (M .EQ. 0 .OR. N .EQ. 0 .OR.
     +    ((ALPHA .EQ. ZERO .OR. K .EQ. 0) .AND. BETA .EQ. ONE)) RETURN
C     C := BETA*C FIRST, THEN ADD THE PRODUCT
      CALL DGESCL(M, N, BETA, C, LDC)
      IF (ALPHA .EQ. ZERO .OR. K .EQ. 0) RETURN
      IF (NOTB) THEN
         IF (NOTA) THEN
C           C := ALPHA*A*B + C
            DO 50 J = 1, N
               DO 40 L = 1, K
                  IF (B(L,J) .NE. ZERO) THEN
                     TEMP = ALPHA*B(L,J)
                     DO 30 I = 1, M
                        C(I,J) = C(I,J) + TEMP*A(I,L)
   30                CONTINUE
                  END IF
   40          CONTINUE
   50       CONTINUE
         ELSE
C           C := ALPHA*A'*B + C
            DO 90 J = 1, N
               DO 80 I = 1, M
                  TEMP = ZERO
                  DO 70 L = 1, K
                     TEMP = TEMP + A(L,I)*B(L,J)
   70             CONTINUE
                  C(I,J) = C(I,J) + ALPHA*TEMP
   80          CONTINUE
   90       CONTINUE
         END IF
      ELSE
         IF (NOTA) THEN
C           C := ALPHA*A*B' + C
            DO 140 J = 1, N
               DO 130 L = 1, K
                  IF (B(J,L) .NE. ZERO) THEN
                     TEMP = ALPHA*B(J,L)
                     DO 120 I = 1, M
                        C(I,J) = C(I,J) + TEMP*A(I,L)
  120                CONTINUE
                  END IF
  130          CONTINUE
  140       CONTINUE
         ELSE
C           C := ALPHA*A'*B' + C
            DO 180 J = 1, N
               DO 170 I = 1, M
                  TEMP = ZERO
                  DO 160 L = 1, K
                     TEMP = TEMP + A(L,I)*B(J,L)
  160             CONTINUE
                  C(I,J) = C(I,J) + ALPHA*TEMP
  170          CONTINUE
  180       CONTINUE
         END IF
      END IF
      RETURN
      END
C
      SUBROUTINE DGESCL(M, N, BETA, C, LDC)
C     C := BETA*C FOR THE M BY N LEADING BLOCK OF C(LDC,N).
C     BETA = 0 STORES ZEROS WITHOUT READING C.
      INTEGER M, N, LDC
      DOUBLE PRECISION BETA
      DOUBLE PRECISION C(LDC,N)
      INTEGER I, J
C
      IF (BETA .EQ. 1.0D0) RETURN
      DO 20 J = 1, N
         DO 10 I = 1, M
            IF (BETA .EQ. 0.0D0) THEN
               C(I,J) = 0.0D0
            ELSE
               C(I,J) = BETA*C(I,J)
            END IF
   10    CONTINUE
   20 CONTINUE
      RETURN
      END
C
      LOGICAL FUNCTION LSAME(CA, CB)
      CHARACTER*1 CA, CB
      INTEGER INTA, INTB
C
      LSAME = CA .EQ. CB
      IF (LSAME) RETURN
      INTA = ICHAR(CA)
      INTB = ICHAR(CB)
      IF (INTA .GE. 97 .AND. INTA .LE. 122) INTA = INTA - 32
      IF (INTB .GE. 97 .AND. INTB .LE. 122) INTB = INTB - 32
      LSAME = INTA .EQ. INTB
      RETURN
      END
