C     NUMWRD - ENGLISH WORDS FOR AN INTEGER, E.G. 21 -> TWENTY-ONE
C
C     LASTNB  POSITION OF THE LAST NON-BLANK CHARACTER (0 IF ALL BLANK)
C     ADDWRD  APPEND A WORD TO A 120 CHARACTER BUFFER, SEPARATED BY ONE
C             BLANK; POS IS THE NEXT FREE POSITION.  A WORD THAT WOULD
C             NOT FIT IN THE BUFFER IS DROPPED.
C     NUMWRD  WORDS FOR -999999999 .. 999999999, NCH = LENGTH OF THE
C             TEXT.  IERR = 1 (AND NO TEXT) WHEN N IS OUT OF RANGE.
C
      INTEGER FUNCTION LASTNB(S)
      CHARACTER*(*) S
      INTEGER I
      DO 10 I = LEN(S), 1, -1
         IF (S(I:I) .NE. ' ') THEN
            LASTNB = I
            RETURN
         END IF
   10 CONTINUE
      LASTNB = 0
      RETURN
      END
C
      SUBROUTINE ADDWRD(BUF, POS, WORD)
      CHARACTER*120 BUF
      INTEGER POS
      CHARACTER*(*) WORD
      INTEGER LASTNB, N, NEED
      N = LASTNB(WORD)
      IF (N .EQ. 0) RETURN
      NEED = N
      IF (POS .GT. 1) NEED = N + 1
      IF (POS + NEED - 1 .GT. 120) RETURN
      IF (POS .GT. 1) THEN
         BUF(POS:POS) = ' '
         POS = POS + 1
      END IF
      BUF(POS:POS+N-1) = WORD(1:N)
      POS = POS + N
      RETURN
      END
C
      SUBROUTINE NUMWRD(N, WORDS, NCH, IERR)
      INTEGER N, NCH, IERR
      CHARACTER*120 WORDS
      CHARACTER*9 ONES(0:19)
      CHARACTER*7 TENS(2:9)
      CHARACTER*8 SCALE(0:2)
      CHARACTER*20 WORD
      INTEGER LASTNB, G(0:2), M, I, H, R, T, U, POS
      DATA ONES /'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX',
     &   'SEVEN', 'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE',
     &   'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN', 'SEVENTEEN',
     &   'EIGHTEEN', 'NINETEEN'/
      DATA TENS /'TWENTY', 'THIRTY', 'FORTY', 'FIFTY', 'SIXTY',
     &   'SEVENTY', 'EIGHTY', 'NINETY'/
      DATA SCALE /' ', 'THOUSAND', 'MILLION'/
      WORDS = ' '
      NCH = 0
      IERR = 0
      IF (IABS(N) .GT. 999999999) THEN
         IERR = 1
         RETURN
      END IF
      POS = 1
      IF (N .LT. 0) CALL ADDWRD(WORDS, POS, 'MINUS')
      IF (N .EQ. 0) CALL ADDWRD(WORDS, POS, ONES(0))
C     SPLIT |N| INTO GROUPS OF THREE DIGITS, G(2) HOLDS THE MILLIONS
      M = IABS(N)
      DO 10 I = 0, 2
         G(I) = MOD(M, 1000)
         M = M / 1000
   10 CONTINUE
      DO 40 I = 2, 0, -1
         IF (G(I) .EQ. 0) GO TO 40
         H = G(I) / 100
         R = MOD(G(I), 100)
         IF (H .GT. 0) THEN
            CALL ADDWRD(WORDS, POS, ONES(H))
            CALL ADDWRD(WORDS, POS, 'HUNDRED')
         END IF
         IF (R .GE. 20) THEN
            T = R / 10
            U = MOD(R, 10)
            IF (U .EQ. 0) THEN
               WORD = TENS(T)
            ELSE
               WORD = TENS(T)(1:LASTNB(TENS(T)))//'-'//ONES(U)
            END IF
            CALL ADDWRD(WORDS, POS, WORD)
         ELSE IF (R .GT. 0) THEN
            CALL ADDWRD(WORDS, POS, ONES(R))
         END IF
         IF (I .GT. 0) CALL ADDWRD(WORDS, POS, SCALE(I))
   40 CONTINUE
      NCH = POS - 1
      RETURN
      END
