C**********************************************************************
C     BASIC SAMPLE STATISTICS IN DOUBLE PRECISION
C
C     MOMENT  MEAN, SAMPLE VARIANCE AND STANDARD DEVIATION
C     DCOVAR  SAMPLE COVARIANCE OF TWO VECTORS
C     MINMAX  EXTREME VALUES WITH THE INDEX OF THEIR FIRST OCCURRENCE
C
C     MOMENT USES THE CORRECTED TWO-PASS ALGORITHM: THE MEAN IS FOUND
C     FIRST, THEN THE SQUARED DEVIATIONS ARE SUMMED, AND A SECOND
C     TERM CORRECTS THE ROUNDING ERROR OF THE MEAN ITSELF.
C
C     IERR = 0  NORMAL
C            1  N .LT. 1, ALL RESULTS ARE SET TO ZERO
C            2  N .EQ. 1, MEAN IS X(1) BUT VARIANCE IS UNDEFINED (ZERO)
C**********************************************************************
      SUBROUTINE MOMENT(N, X, XMEAN, XVAR, XSD, IERR)
      INTEGER N, IERR
      DOUBLE PRECISION X(N), XMEAN, XVAR, XSD
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO = 0.0D0)
      DOUBLE PRECISION S1, S2, D
      INTEGER I
C
      XMEAN = ZERO
      XVAR = ZERO
      XSD = ZERO
      IERR = 0
      IF (N .LT. 1) THEN
         IERR = 1
         RETURN
      END IF
      S1 = ZERO
      DO 10 I = 1, N
         S1 = S1 + X(I)
   10 CONTINUE
      XMEAN = S1 / DBLE(N)
      IF (N .EQ. 1) THEN
         IERR = 2
         RETURN
      END IF
C     SECOND PASS: S1 = SUM OF DEVIATIONS, S2 = SUM OF SQUARES
      S1 = ZERO
      S2 = ZERO
      DO 20 I = 1, N
         D = X(I) - XMEAN
         S1 = S1 + D
         S2 = S2 + D * D
   20 CONTINUE
      XVAR = (S2 - S1 * S1 / DBLE(N)) / DBLE(N - 1)
      XSD = DSQRT(XVAR)
      RETURN
      END
C
C**********************************************************************
C     DCOVAR - SAMPLE COVARIANCE (DIVISOR N-1); ZERO FOR N .LT. 2
C**********************************************************************
      DOUBLE PRECISION FUNCTION DCOVAR(N, X, Y)
      INTEGER N
      DOUBLE PRECISION X(N), Y(N)
      DOUBLE PRECISION SX, SY, SXY
      INTEGER I
C
      DCOVAR = 0.0D0
      IF (N .LT. 2) RETURN
      SX = 0.0D0
      SY = 0.0D0
      DO 10 I = 1, N
         SX = SX + X(I)
         SY = SY + Y(I)
   10 CONTINUE
      SX = SX / DBLE(N)
      SY = SY / DBLE(N)
      SXY = 0.0D0
      DO 20 I = 1, N
         SXY = SXY + (X(I) - SX) * (Y(I) - SY)
   20 CONTINUE
      DCOVAR = SXY / DBLE(N - 1)
      RETURN
      END
C
C**********************************************************************
C     MINMAX - SMALLEST AND LARGEST ELEMENT AND THEIR FIRST POSITIONS.
C     FOR N .LT. 1 ALL FOUR OUTPUTS ARE ZERO.
C**********************************************************************
      SUBROUTINE MINMAX(N, X, XMIN, XMAX, IMIN, IMAX)
      INTEGER N, IMIN, IMAX
      DOUBLE PRECISION X(N), XMIN, XMAX
      INTEGER I
C
      XMIN = 0.0D0
      XMAX = 0.0D0
      IMIN = 0
      IMAX = 0
      IF (N .LT. 1) RETURN
      XMIN = X(1)
      XMAX = X(1)
      IMIN = 1
      IMAX = 1
      DO 10 I = 2, N
         IF (X(I) .LT. XMIN) THEN
            XMIN = X(I)
            IMIN = I
         END IF
         IF (X(I) .GT. XMAX) THEN
            XMAX = X(I)
            IMAX = I
         END IF
   10 CONTINUE
      RETURN
      END
