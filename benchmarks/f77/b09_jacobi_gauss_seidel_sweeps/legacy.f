C     ==================================================================
C     ITSOLV  -  JACOBI AND GAUSS-SEIDEL ITERATION FOR A*X = B
C
C     METHOD  1 = JACOBI, 2 = GAUSS-SEIDEL.
C     X       STARTING GUESS ON ENTRY, LAST ITERATE ON EXIT.
C     XN      WORK ARRAY OF LENGTH N.
C     TOL     STOP WHEN THE LARGEST CHANGE OF ANY COMPONENT IN ONE
C             SWEEP IS NOT ABOVE TOL.  TOL .LE. 0 SELECTS 10 TIMES THE
C             MACHINE PRECISION; THE VALUE USED IS RETURNED IN TOL.
C     ITER    NUMBER OF THE LAST SWEEP.  WHEN THE LIMIT MAXIT IS
C             EXHAUSTED IT IS LEFT AT MAXIT+1.
C     ERR     LARGEST CHANGE IN THE LAST SWEEP.
C     IERR    0 CONVERGED          1 NO CONVERGENCE IN MAXIT SWEEPS
C             2 BAD N OR LDA       3 UNKNOWN METHOD
C             4 ZERO DIAGONAL ELEMENT
C     ==================================================================
      SUBROUTINE ITSOLV(METHOD, A, LDA, N, B, X, XN, TOL, MAXIT,
     +                  ITER, ERR, IERR)
      INTEGER METHOD, LDA, N, MAXIT, ITER, IERR
      DOUBLE PRECISION A(LDA,N), B(N), X(N), XN(N), TOL, ERR
      DOUBLE PRECISION D1MACH, S
      EXTERNAL D1MACH
      INTEGER I, J
C
      IERR = 0
      ITER = 0
      ERR = 0.0D0
      IF (N .LT. 1 .OR. LDA .LT. N) THEN
         IERR = 2
         RETURN
      END IF
      IF (TOL .LE. 0.0D0) TOL = 10.0D0*D1MACH(4)
C     A ZERO ON THE DIAGONAL CANNOT BE DIVIDED BY
      DO 10 I = 1, N
         IF (A(I,I)) 10, 90, 10
   10 CONTINUE
      DO 300 ITER = 1, MAXIT
         GO TO (100, 200), METHOD
C        UNKNOWN METHOD: THE COMPUTED GO TO FALLS THROUGH TO HERE
         IERR = 3
         RETURN
C        JACOBI: EVERY COMPONENT IS BUILT FROM THE OLD ITERATE
  100    CONTINUE
         DO 130 I = 1, N
            S = B(I)
            DO 110 J = 1, N
               IF (J .NE. I) S = S - A(I,J)*X(J)
  110       CONTINUE
            XN(I) = S/A(I,I)
  130    CONTINUE
         ERR = 0.0D0
         DO 140 I = 1, N
            ERR = DMAX1(ERR, DABS(XN(I) - X(I)))
            X(I) = XN(I)
  140    CONTINUE
         GO TO 250
C        GAUSS-SEIDEL: NEW COMPONENTS ARE USED AS SOON AS THEY EXIST
  200    CONTINUE
         ERR = 0.0D0
         DO 230 I = 1, N
            S = B(I)
            DO 210 J = 1, N
               IF (J .NE. I) S = S - A(I,J)*X(J)
  210       CONTINUE
            S = S/A(I,I)
            ERR = DMAX1(ERR, DABS(S - X(I)))
            X(I) = S
  230    CONTINUE
  250    CONTINUE
         IF (ERR - TOL) 400, 400, 300
  300 CONTINUE
C     THE LOOP RAN OUT: ITER IS NOW MAXIT+1
      IERR = 1
      RETURN
  400 CONTINUE
      IERR = 0
      RETURN
   90 CONTINUE
      IERR = 4
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION RESNRM(A, LDA, N, B, X)
C     LARGEST ABSOLUTE ENTRY OF B - A*X.
      INTEGER LDA, N
      DOUBLE PRECISION A(LDA,N), B(N), X(N)
      DOUBLE PRECISION S
      INTEGER I, J
C
      RESNRM = 0.0D0
      DO 20 I = 1, N
         S = B(I)
         DO 10 J = 1, N
            S = S - A(I,J)*X(J)
   10    CONTINUE
         RESNRM = DMAX1(RESNRM, DABS(S))
   20 CONTINUE
      RETURN
      END
