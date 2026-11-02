C**********************************************************************
C     LEAST-SQUARES STRAIGHT LINE  Y = A + B*X
C
C     LINREG  FITS THE LINE AND RETURNS THE SQUARED CORRELATION R2
C     FITTRN  FITS Y AGAINST A TRANSFORMED ABSCISSA F(X), WHERE F IS
C             A FUNCTION SUPPLIED BY THE CALLER
C     RSSQ    RESIDUAL SUM OF SQUARES OF A GIVEN LINE
C
C     IERR = 0  NORMAL
C            1  FEWER THAN TWO POINTS
C            2  ALL ABSCISSAE EQUAL (VERTICAL LINE, NO SLOPE)
C     A, B AND R2 ARE ZERO WHEN IERR .NE. 0.  FOR CONSTANT Y THE FIT IS
C     EXACT AND R2 IS DEFINED AS 1.
C**********************************************************************
      SUBROUTINE LINREG(N, X, Y, A, B, R2, IERR)
      INTEGER N, IERR
      REAL*8 X(N), Y(N), A, B, R2
      REAL*8 SX, SY, SXX, SXY, SYY, XM, YM, DX, DY, TINY, Z, SQ
      PARAMETER (TINY = 1.0D-300)
      INTEGER I
      INTRINSIC DABS
      SQ(Z) = Z * Z
C
      A = 0.0D0
      B = 0.0D0
      R2 = 0.0D0
      IERR = 0
      IF (N .LT. 2) THEN
         IERR = 1
         RETURN
      END IF
C     FIRST PASS: MEANS
      SX = 0.0D0
      SY = 0.0D0
      DO 10 I = 1, N
         SX = SX + X(I)
         SY = SY + Y(I)
   10 CONTINUE
      XM = SX / DBLE(N)
      YM = SY / DBLE(N)
C     SECOND PASS: CENTRED SUMS OF SQUARES AND PRODUCTS
      SXX = 0.0D0
      SXY = 0.0D0
      SYY = 0.0D0
      DO 20 I = 1, N
         DX = X(I) - XM
         DY = Y(I) - YM
         SXX = SXX + SQ(DX)
         SXY = SXY + DX * DY
         SYY = SYY + SQ(DY)
   20 CONTINUE
      IF (DABS(SXX) .LE. TINY) THEN
         IERR = 2
         RETURN
      END IF
      B = SXY / SXX
      A = YM - B * XM
      IF (SYY .LE. TINY) THEN
         R2 = 1.0D0
      ELSE
         R2 = SXY * SXY / (SXX * SYY)
      END IF
      RETURN
      END
C
C     FITTRN - XT IS WORK SPACE FOR THE TRANSFORMED ABSCISSAE
C
      SUBROUTINE FITTRN(N, X, Y, F, XT, A, B, R2, IERR)
      INTEGER N, IERR
      REAL*8 X(N), Y(N), XT(N), A, B, R2, F
      EXTERNAL F
      REAL*8 A0, B0, RR
      INTEGER I, IE
C
      DO 10 I = 1, N
         XT(I) = F(X(I))
   10 CONTINUE
      CALL LINREG(N, XT, Y, A0, B0, RR, IE)
      A = A0
      B = B0
      R2 = RR
      IERR = IE
      RETURN
      END
C
C     RSSQ - SUM OF (Y(I) - A - B*X(I))**2
C
      REAL*8 FUNCTION RSSQ(N, X, Y, A, B)
      INTEGER N
      REAL*8 X(N), Y(N), A, B
      REAL*8 E
      INTEGER I
C
      RSSQ = 0.0D0
      DO 10 I = 1, N
         E = Y(I) - (A + B * X(I))
         RSSQ = RSSQ + E * E
   10 CONTINUE
      RETURN
      END
