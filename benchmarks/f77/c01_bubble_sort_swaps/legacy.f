C**********************************************************************
C     BSORT - BUBBLE SORT WITH EARLY EXIT FOR A REAL VECTOR
C
C     SORTS X(1), ..., X(N) INTO ASCENDING ORDER, IN PLACE.  EVERY
C     PASS BUBBLES THE LARGEST REMAINING ELEMENT TO THE END.  THE
C     POSITION OF THE LAST EXCHANGE OF A PASS BOUNDS THE NEXT PASS,
C     BECAUSE EVERYTHING BEYOND IT IS ALREADY IN ITS FINAL PLACE.  A
C     PASS WITHOUT EXCHANGES ENDS THE SORT.
C
C     ARGUMENTS
C       N      NUMBER OF ELEMENTS (N .LE. 1 GIVES ONE EMPTY PASS)
C       X      VECTOR OF LENGTH N, REPLACED BY THE SORTED VECTOR
C       NSWAP  (OUTPUT) TOTAL NUMBER OF EXCHANGES MADE
C       NPASS  (OUTPUT) NUMBER OF PASSES, THE FINAL CLEAN ONE INCLUDED
C**********************************************************************
      SUBROUTINE BSORT(N, X, NSWAP, NPASS)
      INTEGER N, NSWAP, NPASS
      REAL X(N)
      INTEGER I, LAST, LASTSW
      REAL T
      LOGICAL SWAPPD
C
      NSWAP = 0
      NPASS = 0
      LAST = N
C     ONE PASS PER VISIT TO LABEL 10
   10 CONTINUE
      SWAPPD = .FALSE.
      LASTSW = 0
      NPASS = NPASS + 1
      DO 20 I = 1, LAST - 1
C        IN ORDER: NOTHING TO DO, GO TO THE END OF THE LOOP BODY
         IF (X(I) .LE. X(I+1)) GO TO 20
         T = X(I)
         X(I) = X(I+1)
         X(I+1) = T
         NSWAP = NSWAP + 1
         SWAPPD = .TRUE.
         LASTSW = I
   20 CONTINUE
      IF (.NOT. SWAPPD) RETURN
      LAST = LASTSW
      GO TO 10
      END
C
C**********************************************************************
C     ISSRTD - TRUE WHEN X(1) .LE. X(2) .LE. ... .LE. X(N)
C              (THE SCAN STOPS AT THE FIRST OUT-OF-ORDER PAIR)
C**********************************************************************
      LOGICAL FUNCTION ISSRTD(N, X)
      INTEGER N
      REAL X(N)
      INTEGER I
C
      ISSRTD = .TRUE.
      DO 10 I = 1, N - 1
         IF (X(I) .GT. X(I+1)) THEN
            ISSRTD = .FALSE.
            RETURN
         END IF
   10 CONTINUE
      RETURN
      END
