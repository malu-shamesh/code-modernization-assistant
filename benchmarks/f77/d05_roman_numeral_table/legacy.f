C     ROMAN - ROMAN NUMERALS FOR 1 TO 3999 AND A PRINTED TABLE
C
C     TOROMN  INTEGER TO ROMAN NUMERAL (IERR = 1 OUTSIDE 1..3999)
C     FROMRM  ROMAN NUMERAL TO INTEGER.  ONLY THE CANONICAL FORM IS
C             ACCEPTED: IERR = 2 FOR A BAD CHARACTER OR LENGTH, IERR = 3
C             FOR A NON-CANONICAL NUMERAL SUCH AS IIII OR VX.
C     PRTROM  PRINT A TABLE FOR N = N1, N2, NSTEP
C
      SUBROUTINE TOROMN(N, ROMAN, IERR)
      INTEGER N, IERR
      CHARACTER*16 ROMAN
      INTEGER VAL(13), I, J, K, L, M, POS
      CHARACTER*2 SYM(13)
      DATA VAL /1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1/
      DATA SYM /'M ', 'CM', 'D ', 'CD', 'C ', 'XC', 'L ', 'XL',
     &          'X ', 'IX', 'V ', 'IV', 'I '/
      ROMAN = ' '
      IERR = 0
      IF (N .LT. 1 .OR. N .GT. 3999) THEN
         IERR = 1
         RETURN
      END IF
      M = N
      POS = 1
      DO 20 I = 1, 13
         K = M / VAL(I)
         L = 2
         IF (SYM(I)(2:2) .EQ. ' ') L = 1
         DO 10 J = 1, K
            ROMAN(POS:POS+L-1) = SYM(I)(1:L)
            POS = POS + L
   10    CONTINUE
         M = MOD(M, VAL(I))
   20 CONTINUE
      RETURN
      END
C
      INTEGER FUNCTION FROMRM(ROMAN, IERR)
      CHARACTER*(*) ROMAN
      INTEGER IERR
      CHARACTER*7 DIGITS
      CHARACTER*16 CHECK
      INTEGER DV(7), V(17), I, K, NCH, TOTAL, IE2
      DATA DIGITS /'IVXLCDM'/
      DATA DV /1, 5, 10, 50, 100, 500, 1000/
      IERR = 0
      FROMRM = 0
C     POSITION OF THE LAST NON-BLANK CHARACTER
      NCH = 0
      DO 10 I = LEN(ROMAN), 1, -1
         IF (ROMAN(I:I) .NE. ' ') THEN
            NCH = I
            GO TO 15
         END IF
   10 CONTINUE
   15 IF (NCH .EQ. 0 .OR. NCH .GT. 16) THEN
         IERR = 2
         RETURN
      END IF
      DO 20 I = 1, NCH
         K = INDEX(DIGITS, ROMAN(I:I))
         IF (K .EQ. 0) THEN
            IERR = 2
            RETURN
         END IF
         V(I) = DV(K)
   20 CONTINUE
C     A SMALLER DIGIT IN FRONT OF A LARGER ONE IS SUBTRACTED
      V(NCH+1) = 0
      TOTAL = 0
      DO 30 I = 1, NCH
         IF (V(I) .LT. V(I+1)) THEN
            TOTAL = TOTAL - V(I)
         ELSE
            TOTAL = TOTAL + V(I)
         END IF
   30 CONTINUE
C     ACCEPT THE VALUE ONLY IF IT CONVERTS BACK TO THE SAME NUMERAL
      CALL TOROMN(TOTAL, CHECK, IE2)
      IF (IE2 .NE. 0 .OR. CHECK .NE. ROMAN) THEN
         IERR = 3
         RETURN
      END IF
      FROMRM = TOTAL
      RETURN
      END
C
      SUBROUTINE PRTROM(N1, N2, NSTEP)
      INTEGER N1, N2, NSTEP
      CHARACTER*16 ROMAN
      INTEGER N, IERR
      DO 10 N = N1, N2, NSTEP
         CALL TOROMN(N, ROMAN, IERR)
         IF (IERR .EQ. 0) THEN
            WRITE(6, 100) N, ROMAN
         ELSE
            WRITE(6, 110) N
         END IF
   10 CONTINUE
      RETURN
  100 FORMAT(1X, I5, 2X, A16)
  110 FORMAT(1X, I5, 2X, 'OUT OF RANGE')
      END
