C     CALDAT - CALENDAR ARITHMETIC AND DATE FORMATTING
C
C     ISLEAP  GREGORIAN LEAP YEAR TEST
C     NDAYS   NUMBER OF DAYS IN A MONTH (0 FOR A BAD MONTH)
C     VALDAT  0 IF THE DATE IS VALID, ELSE 1 BAD YEAR, 2 BAD MONTH,
C             3 BAD DAY.  YEARS 1 TO 9999 ARE ACCEPTED.
C     IDOW    DAY OF THE WEEK, 0 = SUNDAY .. 6 = SATURDAY (SAKAMOTO)
C     DOY     DAY OF THE YEAR
C     NEXTDY  ADVANCE A DATE BY ONE DAY
C     FMTISO  FORMAT AS YYYY-MM-DD   (INTERNAL WRITE TO A CHARACTER)
C     FMTLNG  FORMAT AS, E.G., SATURDAY, JANUARY 1, 2000
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
      LOGICAL FUNCTION ISLEAP(IY)
      INTEGER IY
      ISLEAP = (MOD(IY, 4) .EQ. 0 .AND. MOD(IY, 100) .NE. 0)
     &         .OR. MOD(IY, 400) .EQ. 0
      RETURN
      END
C
      INTEGER FUNCTION NDAYS(IY, IM)
      INTEGER IY, IM
      INTEGER MDAYS(12)
      LOGICAL ISLEAP
      DATA MDAYS /31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/
      IF (IM .LT. 1 .OR. IM .GT. 12) THEN
         NDAYS = 0
      ELSE IF (IM .EQ. 2 .AND. ISLEAP(IY)) THEN
         NDAYS = 29
      ELSE
         NDAYS = MDAYS(IM)
      END IF
      RETURN
      END
C
      INTEGER FUNCTION VALDAT(IY, IM, ID)
      INTEGER IY, IM, ID, NDAYS
      VALDAT = 0
      IF (IY .LT. 1 .OR. IY .GT. 9999) THEN
         VALDAT = 1
      ELSE IF (IM .LT. 1 .OR. IM .GT. 12) THEN
         VALDAT = 2
      ELSE IF (ID .LT. 1 .OR. ID .GT. NDAYS(IY, IM)) THEN
         VALDAT = 3
      END IF
      RETURN
      END
C
C     THE DATE MUST BE VALID.  JANUARY AND FEBRUARY BELONG TO THE
C     PREVIOUS YEAR FOR THE LEAP DAY TERMS, WHICH THE TABLE T OF
C     MONTH OFFSETS ALLOWS FOR.
      INTEGER FUNCTION IDOW(IY, IM, ID)
      INTEGER IY, IM, ID
      INTEGER T(12), Y
      DATA T /0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4/
      Y = IY
      IF (IM .LT. 3) Y = Y - 1
      IDOW = MOD(Y + Y/4 - Y/100 + Y/400 + T(IM) + ID, 7)
      RETURN
      END
C
      INTEGER FUNCTION DOY(IY, IM, ID)
      INTEGER IY, IM, ID, NDAYS, M, NTOT
      NTOT = ID
      DO 20 M = 1, IM - 1
         NTOT = NTOT + NDAYS(IY, M)
   20 CONTINUE
      DOY = NTOT
      RETURN
      END
C
      SUBROUTINE NEXTDY(IY, IM, ID)
      INTEGER IY, IM, ID, NDAYS
      ID = ID + 1
      IF (ID .GT. NDAYS(IY, IM)) THEN
         ID = 1
         IM = IM + 1
         IF (IM .GT. 12) THEN
            IM = 1
            IY = IY + 1
         END IF
      END IF
      RETURN
      END
C
      SUBROUTINE FMTISO(IY, IM, ID, STR, IERR)
      INTEGER IY, IM, ID, IERR, VALDAT
      CHARACTER*10 STR
      STR = ' '
      IERR = VALDAT(IY, IM, ID)
      IF (IERR .NE. 0) RETURN
      WRITE(STR, 100) IY, IM, ID
      RETURN
  100 FORMAT(I4.4, '-', I2.2, '-', I2.2)
      END
C
      SUBROUTINE FMTLNG(IY, IM, ID, STR, IERR)
      INTEGER IY, IM, ID, IERR, VALDAT, IDOW, LASTNB, K
      CHARACTER*40 STR
      CHARACTER*9 DAYNM(0:6), MONNM(12)
      DATA DAYNM /'Sunday', 'Monday', 'Tuesday', 'Wednesday',
     &            'Thursday', 'Friday', 'Saturday'/
      DATA MONNM /'January', 'February', 'March', 'April', 'May',
     &            'June', 'July', 'August', 'September', 'October',
     &            'November', 'December'/
      STR = ' '
      IERR = VALDAT(IY, IM, ID)
      IF (IERR .NE. 0) RETURN
      K = IDOW(IY, IM, ID)
      IF (ID .LT. 10) THEN
         WRITE(STR, 100) DAYNM(K)(1:LASTNB(DAYNM(K))),
     &      MONNM(IM)(1:LASTNB(MONNM(IM))), ID, IY
      ELSE
         WRITE(STR, 110) DAYNM(K)(1:LASTNB(DAYNM(K))),
     &      MONNM(IM)(1:LASTNB(MONNM(IM))), ID, IY
      END IF
      RETURN
  100 FORMAT(A, ', ', A, 1X, I1, ', ', I4)
  110 FORMAT(A, ', ', A, 1X, I2, ', ', I4)
      END
