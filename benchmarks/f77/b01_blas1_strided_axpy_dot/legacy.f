C     ==================================================================
C     BLAS1  -  LEVEL 1 VECTOR KERNELS WITH STORAGE STRIDES
C
C     DAXPY  Y := Y + DA*X        DDOT   DOT PRODUCT OF X AND Y
C     DSCAL  X := DA*X            DCOPY  Y := X
C
C     INCX AND INCY ARE THE DISTANCES BETWEEN CONSECUTIVE ELEMENTS IN
C     THE ARRAYS.  A NEGATIVE STRIDE MEANS THAT THE VECTOR IS WALKED
C     BACKWARDS, STARTING AT ELEMENT (1-N)*INC+1 OF THE ARRAY.
C     ==================================================================
      SUBROUTINE DAXPY(N, DA, DX, INCX, DY, INCY)
      INTEGER N, INCX, INCY
      DOUBLE PRECISION DA, DX(*), DY(*)
      INTEGER I, IX, IY
C
      IF (N .LE. 0) RETURN
      IF (DA .EQ. 0.0D0) RETURN
      IX = 1
      IY = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
      DO 10 I = 1, N
         DY(IY) = DY(IY) + DA*DX(IX)
         IX = IX + INCX
         IY = IY + INCY
   10 CONTINUE
      RETURN
      END
C
      SUBROUTINE DCOPY(N, DX, INCX, DY, INCY)
      INTEGER N, INCX, INCY
      DOUBLE PRECISION DX(*), DY(*)
      INTEGER I, IX, IY
C
      IF (N .LE. 0) RETURN
      IX = 1
      IY = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
      DO 10 I = 1, N
         DY(IY) = DX(IX)
         IX = IX + INCX
         IY = IY + INCY
   10 CONTINUE
      RETURN
      END
C
      SUBROUTINE DSCAL(N, DA, DX, INCX)
      INTEGER N, INCX
      DOUBLE PRECISION DA, DX(*)
      INTEGER I, M
C
      IF (N .LE. 0 .OR. INCX .LE. 0) RETURN
      IF (INCX .EQ. 1) GO TO 20
C     NON-UNIT STRIDE
      NINCX = N*INCX
      DO 10 I = 1, NINCX, INCX
         DX(I) = DA*DX(I)
   10 CONTINUE
      RETURN
C     UNIT STRIDE: CLEAN-UP LOOP FIRST, THEN BLOCKS OF FIVE
   20 M = MOD(N,5)
      IF (M .EQ. 0) GO TO 40
      DO 30 I = 1, M
         DX(I) = DA*DX(I)
   30 CONTINUE
      IF (N .LT. 5) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1, N, 5
         DX(I) = DA*DX(I)
         DX(I+1) = DA*DX(I+1)
         DX(I+2) = DA*DX(I+2)
         DX(I+3) = DA*DX(I+3)
         DX(I+4) = DA*DX(I+4)
   50 CONTINUE
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION DDOT(N, DX, INCX, DY, INCY)
      INTEGER N, INCX, INCY
      DOUBLE PRECISION DX(*), DY(*)
      DOUBLE PRECISION DTEMP
      INTEGER I, IX, IY, M, MP1
C
      DDOT = 0.0D0
      DTEMP = 0.0D0
      IF (N .LE. 0) RETURN
      IF (INCX .EQ. 1 .AND. INCY .EQ. 1) GO TO 20
C     GENERAL STRIDES
      IX = 1
      IY = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
      DO 10 I = 1, N
         DTEMP = DTEMP + DX(IX)*DY(IY)
         IX = IX + INCX
         IY = IY + INCY
   10 CONTINUE
      DDOT = DTEMP
      RETURN
C     BOTH STRIDES ARE ONE: CLEAN-UP LOOP, THEN UNROLLED BY FIVE
   20 M = MOD(N,5)
      IF (M .EQ. 0) GO TO 40
      DO 30 I = 1, M
         DTEMP = DTEMP + DX(I)*DY(I)
   30 CONTINUE
      IF (N .LT. 5) GO TO 60
   40 MP1 = M + 1
      DO 50 I = MP1, N, 5
         DTEMP = DTEMP + DX(I)*DY(I) + DX(I+1)*DY(I+1) +
     +           DX(I+2)*DY(I+2) + DX(I+3)*DY(I+3) +
     +           DX(I+4)*DY(I+4)
   50 CONTINUE
   60 DDOT = DTEMP
      RETURN
      END
