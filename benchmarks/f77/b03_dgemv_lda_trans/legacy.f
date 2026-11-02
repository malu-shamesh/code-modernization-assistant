C     ==================================================================
C     DGEMV  -  MATRIX-VECTOR PRODUCT  Y := ALPHA*OP(A)*X + BETA*Y
C
C     OP(A) IS A OR A-TRANSPOSE (TRANS = 'N', 'T' OR 'C').  A IS STORED
C     COLUMN BY COLUMN IN AN ARRAY WITH LEADING DIMENSION LDA, SO
C     ELEMENT (I,J) IS A(I,J) AND ROWS M+1..LDA ARE NEVER TOUCHED.
C     X AND Y ARE STRIDED (INCX, INCY).  INFO IS ZERO ON SUCCESS,
C     OTHERWISE THE POSITION OF THE FIRST ILLEGAL ARGUMENT.
C     ==================================================================
      SUBROUTINE DGEMV(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA,
     +                 Y, INCY, INFO)
      CHARACTER*1 TRANS
      INTEGER M, N, LDA, INCX, INCY, INFO
      DOUBLE PRECISION ALPHA, BETA
      DOUBLE PRECISION A(LDA,*), X(*), Y(*)
      DOUBLE PRECISION ONE, ZERO
      PARAMETER (ONE = 1.0D0, ZERO = 0.0D0)
      DOUBLE PRECISION TEMP
      INTEGER I, IX, IY, J, JX, JY, KX, KY, LENX, LENY
      LOGICAL LSAME
      EXTERNAL LSAME
C
      INFO = 0
      IF (.NOT.LSAME(TRANS,'N') .AND. .NOT.LSAME(TRANS,'T') .AND.
     +    .NOT.LSAME(TRANS,'C')) THEN
         INFO = 1
      ELSE IF (M .LT. 0) THEN
         INFO = 2
      ELSE IF (N .LT. 0) THEN
         INFO = 3
      ELSE IF (LDA .LT. MAX(1,M)) THEN
         INFO = 6
      ELSE IF (INCX .EQ. 0) THEN
         INFO = 8
      ELSE IF (INCY .EQ. 0) THEN
         INFO = 11
      END IF
      IF (INFO .NE. 0) RETURN
C     QUICK RETURN WHEN THERE IS NOTHING TO DO
      IF (M .EQ. 0 .OR. N .EQ. 0 .OR.
     +    (ALPHA .EQ. ZERO .AND. BETA .EQ. ONE)) RETURN
C     LENGTHS OF X AND Y, AND START POSITIONS FOR NEGATIVE STRIDES
      IF (LSAME(TRANS,'N')) THEN
         LENX = N
         LENY = M
      ELSE
         LENX = M
         LENY = N
      END IF
      KX = 1
      IF (INCX .LT. 0) KX = 1 - (LENX-1)*INCX
      KY = 1
      IF (INCY .LT. 0) KY = 1 - (LENY-1)*INCY
C     FIRST FORM Y := BETA*Y.  BETA = 0 MEANS Y IS NOT READ.
      IF (BETA .NE. ONE) THEN
         IY = KY
         DO 10 I = 1, LENY
            IF (BETA .EQ. ZERO) THEN
               Y(IY) = ZERO
            ELSE
               Y(IY) = BETA*Y(IY)
            END IF
            IY = IY + INCY
   10    CONTINUE
      END IF
      IF (ALPHA .EQ. ZERO) RETURN
      IF (LSAME(TRANS,'N')) THEN
C        Y := Y + ALPHA*A*X, ACCUMULATED COLUMN BY COLUMN
         JX = KX
         DO 40 J = 1, N
            IF (X(JX) .NE. ZERO) THEN
               TEMP = ALPHA*X(JX)
               IY = KY
               DO 30 I = 1, M
                  Y(IY) = Y(IY) + TEMP*A(I,J)
                  IY = IY + INCY
   30          CONTINUE
            END IF
            JX = JX + INCX
   40    CONTINUE
      ELSE
C        Y := Y + ALPHA*A'*X, ONE DOT PRODUCT PER COLUMN OF A
         JY = KY
         DO 60 J = 1, N
            TEMP = ZERO
            IX = KX
            DO 50 I = 1, M
               TEMP = TEMP + A(I,J)*X(IX)
               IX = IX + INCX
   50       CONTINUE
            Y(JY) = Y(JY) + ALPHA*TEMP
            JY = JY + INCY
   60    CONTINUE
      END IF
      RETURN
      END
C
      LOGICAL FUNCTION LSAME(CA, CB)
C     TRUE IF CA AND CB ARE THE SAME LETTER, IGNORING CASE (ASCII).
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
