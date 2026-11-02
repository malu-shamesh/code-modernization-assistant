C**********************************************************************
C     ORDER STATISTICS OF A REAL SAMPLE
C
C     COPYSR  COPIES X INTO THE WORK VECTOR W AND SORTS W ASCENDING
C     XMEDN   MEDIAN (MEAN OF THE TWO MIDDLE VALUES FOR EVEN N)
C     XPCTL   PERCENTILE, LINEAR INTERPOLATION BETWEEN ORDER STATISTICS
C     NEARRK  NEAREST-RANK POSITION OF A PERCENTILE
C     RANKS   RANKS WITH TIES REPLACED BY THEIR AVERAGE RANK
C
C     THE INPUT VECTOR X IS NEVER MODIFIED.
C**********************************************************************
      SUBROUTINE COPYSR(N, X, W)
      INTEGER N
      REAL X(N), W(N)
      INTEGER I, J
      REAL T
      DO 10 I = 1, N
         W(I) = X(I)
   10 CONTINUE
      DO 40 I = 2, N
         T = W(I)
         J = I - 1
   20    IF (J .LT. 1) GO TO 30
         IF (W(J) .LE. T) GO TO 30
         W(J+1) = W(J)
         J = J - 1
         GO TO 20
   30    W(J+1) = T
   40 CONTINUE
      RETURN
      END
C
      REAL FUNCTION XMEDN(N, X, W)
      INTEGER N
      REAL X(N), W(N)
      XMEDN = 0.0
      IF (N .LT. 1) RETURN
      CALL COPYSR(N, X, W)
      IF (MOD(N, 2) .EQ. 1) THEN
         XMEDN = W((N + 1) / 2)
      ELSE
         XMEDN = 0.5 * (W(N / 2) + W(N / 2 + 1))
      END IF
      RETURN
      END
C
C     XPCTL - P IS IN PERCENT AND IS CLAMPED TO 0..100.  THE 1-BASED
C     POSITION IS H = (N-1)*P/100 + 1; THE RESULT INTERPOLATES BETWEEN
C     W(INT(H)) AND THE NEXT ORDER STATISTIC.
C
      REAL FUNCTION XPCTL(N, X, W, P)
      INTEGER N
      REAL X(N), W(N), P
      REAL PP, H, F, HPOS
      INTEGER K
      HPOS(Q) = FLOAT(N - 1) * Q / 100.0 + 1.0
      XPCTL = 0.0
      IF (N .LT. 1) RETURN
      CALL COPYSR(N, X, W)
      PP = AMAX1(0.0, AMIN1(100.0, P))
      H = HPOS(PP)
      K = INT(H)
      IF (K .GE. N) THEN
         XPCTL = W(N)
      ELSE
         F = H - FLOAT(K)
         XPCTL = W(K) + F * (W(K+1) - W(K))
      END IF
      RETURN
      END
C
C     NEARRK - RANK OF THE P-TH PERCENTILE BY THE NEAREST-RANK RULE,
C     ROUNDED HALF AWAY FROM ZERO AND KEPT INSIDE 1..N (N .GE. 1)
C
      INTEGER FUNCTION NEARRK(N, P)
      INTEGER N
      REAL P
      NEARRK = NINT(P * FLOAT(N) / 100.0)
      NEARRK = MAX0(1, MIN0(N, NEARRK))
      RETURN
      END
C
C     RANKS - R(I) IS THE RANK OF X(I) (1 = SMALLEST).  IDX RECEIVES
C     THE STABLE SORT PERMUTATION; NTIE COUNTS THE GROUPS OF TIES.
C
      SUBROUTINE RANKS(N, X, R, IDX, NTIE)
      INTEGER N, IDX(N), NTIE
      REAL X(N), R(N)
      INTEGER I, J, K, L
      REAL AVG
      NTIE = 0
      DO 10 I = 1, N
         IDX(I) = I
   10 CONTINUE
      DO 40 I = 2, N
         K = IDX(I)
         J = I - 1
   20    IF (J .LT. 1) GO TO 30
         IF (X(IDX(J)) .LE. X(K)) GO TO 30
         IDX(J+1) = IDX(J)
         J = J - 1
         GO TO 20
   30    IDX(J+1) = K
   40 CONTINUE
C     SORTED POSITIONS I..J FORM ONE GROUP OF EQUAL VALUES
      I = 1
   50 IF (I .GT. N) RETURN
      J = I
   60 IF (J .GE. N) GO TO 70
      IF (X(IDX(J+1)) .NE. X(IDX(I))) GO TO 70
      J = J + 1
      GO TO 60
   70 AVG = 0.5 * FLOAT(I + J)
      IF (J .GT. I) NTIE = NTIE + 1
      DO 80 L = I, J
         R(IDX(L)) = AVG
   80 CONTINUE
      I = J + 1
      GO TO 50
      END
