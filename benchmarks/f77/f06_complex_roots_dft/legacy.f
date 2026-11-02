C     CPLXRT - QUADRATIC ROOTS, A SMALL DFT AND A CALL STATISTIC
C
C     QUADRT SOLVES A*Z**2 + B*Z + C = 0 FOR REAL COEFFICIENTS AND
C     ALWAYS RETURNS TWO COMPLEX ROOTS.  THE NUMBER OF CALLS AND OF
C     COMPLEX-PAIR RESULTS ARE KEPT IN COMMON /QSTAT/, WHICH BLOCK
C     DATA QSTBD STARTS AT ZERO.
C
      BLOCK DATA QSTBD
      COMMON /QSTAT/ NCALL, NCPLX
      DATA NCALL /0/, NCPLX /0/
      END
C
C     IKIND  0 TWO REAL ROOTS    1 DOUBLE REAL ROOT
C            2 COMPLEX PAIR      3 LINEAR (A = 0, ONE ROOT IN Z1)
C           -1 NO SOLUTION OR EVERY Z IS A SOLUTION (A = B = 0)
      SUBROUTINE QUADRT(A, B, C, Z1, Z2, IKIND)
      REAL A, B, C
      COMPLEX Z1, Z2
      INTEGER IKIND
      COMMON /QSTAT/ NCALL, NCPLX
      NCALL = NCALL + 1
      Z1 = (0.0, 0.0)
      Z2 = (0.0, 0.0)
      IF (A .EQ. 0.0) THEN
         IF (B .EQ. 0.0) THEN
            IKIND = -1
         ELSE
            IKIND = 3
            Z1 = CMPLX(-C / B, 0.0)
         END IF
         RETURN
      END IF
      D = B*B - 4.0*A*C
      IF (D .GT. 0.0) THEN
C        AVOID CANCELLATION: Q HAS THE SAME SIGN AS B
         Q = -0.5 * (B + SIGN(SQRT(D), B))
         Z1 = CMPLX(Q / A, 0.0)
         Z2 = CMPLX(C / Q, 0.0)
         IKIND = 0
      ELSE IF (D .EQ. 0.0) THEN
         Z1 = CMPLX(-B / (2.0*A), 0.0)
         Z2 = Z1
         IKIND = 1
      ELSE
         RE = -B / (2.0*A)
         AIM = SQRT(-D) / (2.0*A)
         Z1 = CMPLX(RE, ABS(AIM))
         Z2 = CONJG(Z1)
         IKIND = 2
         NCPLX = NCPLX + 1
      END IF
      RETURN
      END
C
C     SAME EQUATION WITH COMPLEX COEFFICIENTS.  IERR = 1 WHEN CA = 0.
      SUBROUTINE CQUAD(CA, CB, CC, Z1, Z2, IERR)
      COMPLEX CA, CB, CC, Z1, Z2, D, S
      INTEGER IERR
      IERR = 0
      Z1 = (0.0, 0.0)
      Z2 = (0.0, 0.0)
      IF (CA .EQ. (0.0, 0.0)) THEN
         IERR = 1
         RETURN
      END IF
      D = CB*CB - 4.0*CA*CC
      S = CSQRT(D)
      Z1 = (-CB + S) / (2.0*CA)
      Z2 = (-CB - S) / (2.0*CA)
      RETURN
      END
C
C     DISCRETE FOURIER TRANSFORM OF X(1..N) INTO XF(1..N).
C     ISIGN = -1 FORWARD, +1 INVERSE (INCLUDING THE FACTOR 1/N).
      SUBROUTINE CDFT(N, X, XF, ISIGN)
      INTEGER N, ISIGN
      COMPLEX X(*), XF(*), S
      PI2 = 6.283185307
      DO 20 K = 0, N-1
         S = (0.0, 0.0)
         DO 10 J = 0, N-1
            ANG = FLOAT(ISIGN) * PI2 * FLOAT(MOD(J*K, N)) / FLOAT(N)
            S = S + X(J+1) * CMPLX(COS(ANG), SIN(ANG))
   10    CONTINUE
         IF (ISIGN .GT. 0) S = S / FLOAT(N)
         XF(K+1) = S
   20 CONTINUE
      RETURN
      END
C
C     TOTAL POWER OF X(1..N): SUM OF REAL**2 + AIMAG**2
      REAL FUNCTION CPOWER(N, X)
      INTEGER N
      COMPLEX X(*), Z
      REAL ZSQ
      ZSQ(Z) = REAL(Z)**2 + AIMAG(Z)**2
      P = 0.0
      DO 30 I = 1, N
         P = P + ZSQ(X(I))
   30 CONTINUE
      CPOWER = P
      RETURN
      END
C
C     RETURN AND OPTIONALLY CLEAR THE CALL STATISTICS
      SUBROUTINE QSTATS(NC, NX, IRESET)
      COMMON /QSTAT/ NCALL, NCPLX
      NC = NCALL
      NX = NCPLX
      IF (IRESET .NE. 0) THEN
         NCALL = 0
         NCPLX = 0
      END IF
      RETURN
      END
