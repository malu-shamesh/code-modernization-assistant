C     ==================================================================
C     TRNSPK  -  IN-PLACE TRANSPOSES AND PACKED TRIANGULAR STORAGE
C
C     DSWAP   EXCHANGES TWO STRIDED VECTORS.
C     SQTRNS  TRANSPOSES A SQUARE N BY N MATRIX IN A(LDA,*): THE PART
C             OF COLUMN J BELOW THE DIAGONAL IS SWAPPED WITH THE PART
C             OF ROW J RIGHT OF THE DIAGONAL, BY PASSING THE ARRAY
C             ELEMENTS A(J+1,J) AND A(J,J+1) TO DSWAP.
C     RCTRNS  TRANSPOSES AN M BY N MATRIX HELD COLUMN BY COLUMN IN THE
C             VECTOR A(*); THE RESULT IS N BY M IN THE SAME VECTOR.  IT
C             FOLLOWS THE CYCLES OF THE PERMUTATION.  MARK(*) IS WORK
C             SPACE OF LENGTH M*N SUPPLIED BY THE CALLER.
C     PACKUP  COPIES THE UPPER TRIANGLE OF A INTO AP(*) COLUMN BY
C             COLUMN: ELEMENT (I,J) GOES TO AP(I+J*(J-1)/2).
C     UNPKUP  THE INVERSE OF PACKUP.  A BECOMES SYMMETRIC.
C     ==================================================================
      SUBROUTINE DSWAP(N, DX, INCX, DY, INCY)
      INTEGER N, INCX, INCY
      DOUBLE PRECISION DX(*), DY(*)
      DOUBLE PRECISION DTEMP
      INTEGER I, IX, IY
C
      IF (N .LE. 0) RETURN
      IX = 1
      IY = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
      DO 10 I = 1, N
         DTEMP = DX(IX)
         DX(IX) = DY(IY)
         DY(IY) = DTEMP
         IX = IX + INCX
         IY = IY + INCY
   10 CONTINUE
      RETURN
      END
C
      SUBROUTINE SQTRNS(A, LDA, N)
      INTEGER LDA, N
      DOUBLE PRECISION A(LDA,*)
      INTEGER J
C
      DO 10 J = 1, N-1
         CALL DSWAP(N-J, A(J+1,J), 1, A(J,J+1), LDA)
   10 CONTINUE
      RETURN
      END
C
      SUBROUTINE RCTRNS(A, M, N, MARK)
      INTEGER M, N
      DOUBLE PRECISION A(*)
      LOGICAL MARK(*)
      DOUBLE PRECISION T, T2
      INTEGER I, J, K
C
      IF (M .LE. 1 .OR. N .LE. 1) RETURN
      MN = M*N
      MN1 = MN - 1
      DO 10 K = 1, MN
         MARK(K) = .FALSE.
   10 CONTINUE
C     THE FIRST AND THE LAST ELEMENT NEVER MOVE
      MARK(1) = .TRUE.
      MARK(MN) = .TRUE.
      DO 40 K = 2, MN1
         IF (MARK(K)) GO TO 40
C        WALK THE CYCLE THAT STARTS AT POSITION K.  THE ELEMENT NOW AT
C        POSITION I BELONGS AT POSITION (I-1)*N MOD (M*N-1) + 1.
         I = K
         T = A(K)
   20    CONTINUE
            J = MOD((I-1)*N, MN1) + 1
            T2 = A(J)
            A(J) = T
            T = T2
            MARK(J) = .TRUE.
            I = J
         IF (I .NE. K) GO TO 20
   40 CONTINUE
      RETURN
      END
C
      SUBROUTINE PACKUP(A, LDA, N, AP)
      INTEGER LDA, N
      DOUBLE PRECISION A(LDA,N), AP(*)
      INTEGER I, J
C     POSITION OF ELEMENT (I,J), I .LE. J, IN PACKED STORAGE
      KUP(I,J) = I + (J*(J-1))/2
C
      DO 20 J = 1, N
         DO 10 I = 1, J
            AP(KUP(I,J)) = A(I,J)
   10    CONTINUE
   20 CONTINUE
      RETURN
      END
C
      SUBROUTINE UNPKUP(AP, A, LDA, N)
      INTEGER LDA, N
      DOUBLE PRECISION AP(*), A(LDA,N)
      INTEGER I, J
      KUP(I,J) = I + (J*(J-1))/2
C
      DO 20 J = 1, N
         DO 10 I = 1, J
            A(I,J) = AP(KUP(I,J))
            A(J,I) = AP(KUP(I,J))
   10    CONTINUE
   20 CONTINUE
      RETURN
      END
