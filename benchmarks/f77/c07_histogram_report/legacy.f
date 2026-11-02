C**********************************************************************
C     HISTOGRAM OF A REAL SAMPLE WITH A PRINTED REPORT
C
C     HISTO   COUNTS THE VALUES OF X INTO NB EQUAL-WIDTH BINS ON
C             [XMIN, XMAX].  A VALUE EQUAL TO XMAX BELONGS TO THE LAST
C             BIN.  NOUT RETURNS THE NUMBER OF VALUES OUTSIDE THE RANGE,
C             OR -1 (ICNT UNTOUCHED) WHEN NB .LT. 1 OR XMAX .LE. XMIN.
C     PRTHST  PRINTS ONE LINE PER BIN WITH A BAR OF UP TO TEN STARS
C             AND A TOTALS LINE.
C     HISTDR  DRIVER PROGRAM WITH BUILT-IN SAMPLE DATA.
C**********************************************************************
      SUBROUTINE HISTO(N, X, XMIN, XMAX, NB, ICNT, NOUT)
      INTEGER N, NB, NOUT, ICNT(NB)
      REAL X(N), XMIN, XMAX
      REAL W
      INTEGER I, K
C
      IF (NB .LT. 1 .OR. XMAX .LE. XMIN) THEN
         NOUT = -1
         RETURN
      END IF
      W = (XMAX - XMIN) / FLOAT(NB)
      NOUT = 0
      DO 10 K = 1, NB
         ICNT(K) = 0
   10 CONTINUE
      DO 20 I = 1, N
         IF (X(I) .LT. XMIN .OR. X(I) .GT. XMAX) THEN
            NOUT = NOUT + 1
         ELSE
C           TRUNCATION IS SAFE HERE: THE OFFSET IS NEVER NEGATIVE
            K = INT((X(I) - XMIN) / W) + 1
            IF (K .GT. NB) K = NB
            ICNT(K) = ICNT(K) + 1
         END IF
   20 CONTINUE
      RETURN
      END
C
C**********************************************************************
C     PRTHST - THE BAR LENGTH IS 10*COUNT/MAXCOUNT ROUNDED BY NINT
C**********************************************************************
      SUBROUTINE PRTHST(NB, ICNT, XMIN, XMAX, NOUT)
      INTEGER NB, ICNT(NB), NOUT
      REAL XMIN, XMAX
      CHARACTER*10 BAR
      REAL W, XLO
      INTEGER K, J, L, MAXC, NTOT
C
      IF (NB .LT. 1) THEN
         WRITE (*, *) 'NO BINS'
         RETURN
      END IF
      W = (XMAX - XMIN) / FLOAT(NB)
      MAXC = 0
      NTOT = 0
      DO 10 K = 1, NB
         MAXC = MAX0(MAXC, ICNT(K))
         NTOT = NTOT + ICNT(K)
   10 CONTINUE
      WRITE (*, 100)
      DO 40 K = 1, NB
         L = 0
         IF (MAXC .GT. 0) L = NINT(10.0 * FLOAT(ICNT(K)) / FLOAT(MAXC))
         BAR = ' '
         DO 20 J = 1, L
            BAR(J:J) = '*'
   20    CONTINUE
         XLO = XMIN + FLOAT(K - 1) * W
         WRITE (*, 110) K, XLO, XLO + W, BAR, ICNT(K)
   40 CONTINUE
      WRITE (*, 120) NTOT, NOUT
      RETURN
  100 FORMAT (1X, 3HBIN, 5X, 3HLOW, 4X, 4HHIGH, 2X, 1H|, 10X, 1H|,
     +        5HCOUNT)
  110 FORMAT (1X, I3, 2F8.2, 2X, 1H|, A10, 1H|, I5)
  120 FORMAT (1X, 6HTOTAL=, I6, 2X, 9HOUTSIDE =, I5)
      END
C
C**********************************************************************
C     HISTDR - TWELVE SAMPLE VALUES IN FIVE BINS ON [0, 10]
C**********************************************************************
      PROGRAM HISTDR
      INTEGER ICNT(5), NOUT
      REAL X(12)
      DATA X /0.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 9.0,
     +        10.0, 11.0/
      CALL HISTO(12, X, 0.0, 10.0, 5, ICNT, NOUT)
      CALL PRTHST(5, ICNT, 0.0, 10.0, NOUT)
      END
