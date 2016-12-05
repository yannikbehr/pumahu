C ===========================================================================
      PROGRAM M2011
C ===========================================================================
C
C	A program to run calculations of heat and mass balances of Crater Lake,
C	given various assumptions. Based on program LAKE (A W Hurst, 1979)
C	Input lake temperatures &  chemical concentrations, with dilution
C	Use standard wind velocity, or read input (e.g. based on Chateau/Waiouru
C	Starts with dilution of Mg and Cl, then analyses water balances
C	Looks at chloride content of steam
C	This version for lake after 2003
!       Different units, more suitable for daily inputs
C	Lake area taken from absolute height
C	NO Plots on screen or laserprinter, ONLY prints listing
C	Adding Evaporation & Meltflow to 350 day Chloride Calculation
C ---------------------------------------------------------------------------
!       NEW SECTION stable isotopes
!       O18 and H2 are measured in deviation in ppt (1/1000) from normal
!       Bruce Christenson's Formulae for O18 & H2, maybe now correct
!       Some items fixed for M2011, eg H2 for melt
	IMPLICIT NONE
	COMMON /L/ A,VOL
!	COMMON /PLOT/ TSCALE,YEARI
!	COMMON /CONST/ P0,P1,P2,E0,E1,E2
!	PARAMETER (TFIX = 610)
	INTEGER*8 itime
	CHARACTER     COMMENT
	CHARACTER*3   CVAL
	CHARACTER*40  CAPTION,cctime
	CHARACTER*130 CBUFFER
	INTEGER YR,MO,DY,MM,NSTART,NFINISH,NPREV
	INTEGER ND(10000)
	INTEGER*4 N,NN,NO
	INTEGER NCL(200)
	INTEGER ICL,IMG,NCLAV,CLDAY
C  Input Lake Variables 	
	REAL C(10000),DV(10000),F(10000),H(10000),M(10000),T(10000) 
C  Input Wind Variables
	REAL W(10000),X(10000),O18(10000),O18M(10000),H2(10000),H2M(10000)
	REAL INF,MASS,MASSP,MELTF,MG,MGT(10000),AVCL(200),Oheavy,Deut,O18F,H2F
	REAL A,VOL,VOLP,LOSS,TIMEM,FACTOR
	REAL ENTHAL,WIND,E,EL,EV,TE,HGT,FL,DR,DRCL,DRMG,DENSITY
	REAL STEAM,EVAP,STFL,EVFL,FMELT,PWR
	REAL CLTOT,EVTOT,MTTOT,STTOT,TTTOT,MGTOT
	REAL FCL(10000),FMG(10000),CLT(10000),P(10000)
	REAL STCL(400),TCL(400),TEV(400),TMT(400),TAV(400),TMG(400)

	OPEN(UNIT=7,FILE='input.dat')			! Copy of data.dat
	OPEN(UNIT=9,FILE='data.dat',STATUS='OLD')	! Input file

C-- Initial Setup
	ENTHAL=6.0		!     TERAJOULES/GIGAGRAM (=000 M3 WATER)
!   Day 1 is 1/1/2000
	MM = 10000		!  Max number of days to calculate

C-- Default wind velocity 
	Print *,' Default Wind Velocity m/s [2.0]: '
	READ(5,1000) WIND
1000	FORMAT(2F3.0)

!  Enter default wind for every day
	DO N = 1,MM
	   W(N) = WIND
	END DO 

!	Test Fractionation
	DO N = 0,60,10
	   T(1) = FLOAT(N)
	   CALL FRACTIONATE(T(1),O18F,H2F)
	   PRINT *,'Temp  ',T(1),'  O18 Factor  ',O18F,'  H2 Factor  ', H2F
	END DO


C  Read WIND.BAT file if it exists
	OPEN(UNIT=8,ERR=50,FILE='wind.dat',STATUS='OLD')
	DO N = 1,MM
	   READ(8,*,END=50) YR,MO,DY,WIND
	   CALL DAYNO(YR,MO,DY,NO)
	   W(NO) = WIND
	END DO 
	PRINT *,'WIND.BAT file has been read'
   50	CONTINUE

!	OPEN(6,FILE='LAKEOUT.DAT',STATUS='NEW')
	OPEN(6,FILE='LAKEOUTm.DAT')
	CAPTION = ' '
	itime = time8()
	call ctime(itime,cctime)
	print *, 'Program run at  ', cctime
!	print *, 'Following lines are input file NEW.DAT'
!	WRITE(6,289)
  289	FORMAT('  YEAR  MO  DY  TEMP Lvl    Flow  Mg    Cl  Dil ')




C-- Read Data
        N=1
  100	READ(9,109,END=170)  COMMENT,CBUFFER
  109	FORMAT(A1,A130)
	IF(COMMENT .NE. ' ') GOTO 100	! Allows comment lines
	READ(CBUFFER,*) YR,MO,DY,TE,HGT,FL,IMG,ICL,DR,Oheavy,Deut
	CALL DAYNO(YR,MO,DY,NO)
!	write(7,2198) yr,mo,dy,te,hgt,fl,img/1.,icl/1.,dr,no,wind,O18(n),H2(n)
	IF(N .EQ. 1) THEN
	   ND(NO) = NO
	   NSTART = NO
	   NPREV  = NO
	   T(NO)=TE			! Temperature
	   H(NO)=HGT
	   F(NO)=FL
	   O18(NO)=Oheavy		! O18 value
	   H2(NO)=Deut			! Deuterium
	   O18M(NO)=Oheavy		! O18 initial value for model
	   H2M(NO)=Deut			! Deuterium initial value for model
	   M(NO)=IMG/1000.		! Magnesium content	parts per 1000
	   C(NO)=ICL/1000.		! Chloride content	parts per 1000
	   DV(NO) = 1.0
	END IF
	NFINISH = NO

C  No value for DR will be read as 1.0
	IF(DR .LT. 0.1) DR = 1.0
!	DV(NO)=DR		! Dilution ratio (cf 1.0 for no dilution)
!  Interpolation here for Temperature 
	DO NN = NPREV+1,NFINISH
	   T(NN)=TE+(T(NPREV)-TE)*(NO-NN)/(NO-NPREV)			! Temperature
	   O18(NN)=Oheavy+(O18(NPREV)-Oheavy)*(NO-NN)/(NO-NPREV)	! O18
	   H2(NN)=Deut+(H2(NPREV)-Deut)*(NO-NN)/(NO-NPREV)		! Deuterium
	   M(NN)=IMG/1000. + (M(NPREV)-IMG/1000.)*(NO-NN)/(NO-NPREV)		! Mg  parts per 1000
	   C(NN)=ICL/1000. + (C(NPREV)-ICL/1000.)*(NO-NN)/(NO-NPREV)		! Cl  parts per 1000
	   DV(NN) = 1.0 + (DR-1.0)/(NO-NPREV)					! Dilution shared out 
	   ND(NN) = NN
	   H(NN)=HGT+(H(NPREV)-HGT)*(NO-NN)/(NO-NPREV)			! Level
	   F(NN) = F(NPREV)						! Flow (or level), same as last reading 
	END DO
	CALL DAYNO(YR,MO,DY,NPREV)

	F(NO)=FL			! Flow 
!	ND(NO)=NO		! ND(N) is array of Daynumbers
  	N=N+1
	GO TO 100

C-- End of Data, start Calculation
  170	Continue
	print *, NSTART,' to ',NFINISH,' days from 1/1/2000.'
	write(7,*) NSTART,' to ',NFINISH,' days from 1/1/2000.'
	DO N = NSTART,NFINISH
	   CALL DATEF(YR,MO,DY,N)
	   write(7,2198) yr,mo,dy,t(n),h(n),f(n),m(n),c(n),dv(n),nd(n),w(n),O18(n),H2(n)
 2198   format(i6,2i4,f6.1,f8.2,f6.1,2f6.3,f6.2,i9,f5.1,2f7.2)
	END DO
C FULLNESS calculates VOL & A(rea) from Lake Level

	CALL FULLNESS(H(NSTART))     
	DENSITY = 1.003 - 0.00033*T(NSTART)	! Density about 1.0
	MASS = VOL * DENSITY		! MASS of water in KT

C-- Since M(N),C(N) in 1/1000, masses in T
	MGT(NSTART)=M(NSTART)*MASS		! Initial total mass Mg	T
	CLT(NSTART)=C(NSTART)*MASS		! Initial total mass Cl	T
!	NPLOT = 0
	NCLAV = 0
	CLTOT = 0.0
	MGTOT = 0.0
	EVTOT = 0.0
	MTTOT = 0.0
	STTOT = 0.0
	TTTOT = 0.0
	CLDAY = 0.0
	WRITE(6,*) 'Crater Lake Balances   ',CAPTION
	WRITE(6,299)
  299	FORMAT('  YR MO DY   TEMP    STFL     PWR      EVFL    FMELT    INF   (DR-1)*M',
     *	'   M-MP   FMG    FCL   O18    O18 M   O18F     H2    H2  M')
C-- Start Main Loop
	DO N = NSTART+1,NFINISH

C-- Calculate Year, Month, Day for Printout
	    CALL DATEF(YR,MO,DY,ND(N))
!	    print *, 'More Dates  ', yr, mo, dy, nd(n)
!	    YR=YR-1900
	    TIMEM =.0864*(ND(N)-ND(N-1))	! Time Interval in Megaseconds
!   Will not generally use TIMEM except for Power
	    IF(M(N).LT.0.1) M(N)=M(N-1)	     ! Mg not measured, use last value
	    IF(C(N).LT.0.1) C(N)=C(N-1)	     ! Cl not measured, use last value
	    DENSITY = 1.003 - 0.00033*T(N)

C-- VOLUME and MASS calculations, also A(rea) depends on volume
	    VOLP = VOL			! Previous VOL 
	    MASSP = MASS		! Previous MASS
	    CALL FULLNESS(H(N))     
	    MASS = VOL * DENSITY	! in KT
!   Next line is to check Volume & Area calculations OK
!   May add Wind checking here later when on 1-day version
!	    write(6,1198) yr,mo,dy,t(n),f(n),m(n),c(n),dv(n),VOL,A,TIMEM
 1198   format(i6,2i4,f6.1,f8.2,2f6.3,f6.2,f11.0,f9.0,f9.4)

C-- DR is dilution ratio	! 1.0 for none, >1.0 if lake was diluted
	    DR = DV(N)		! Dilution due to loss of water

C-- Mass balances for Mg and Cl
	    MGT(N)=MGT(N-1) + MASS*M(N) - MASSP*M(N-1)/DR   ! New total Mg input
	    FMG(N)=(MGT(N) - MGT(N-1))/(ND(N)-ND(N-1))	    ! Mg input T/day
	    IF((MGT(N-1)-MGT(N)).GT. 0.02*MASS*M(N)) THEN
		DRMG = 0.98 * MASSP * M(N-1)/(MASS * M(N))
!		WRITE(6,1120) YR,MO,DY,DR,DRMG
	    END IF
1120	    FORMAT(' At ',3I4,' Mg dilution query, change from',F6.3,' to',F6.3)

	    CLT(N)=CLT(N-1) + MASS*C(N) - MASSP*C(N-1)/DR   ! New total Cl input
	    FCL(N)=(CLT(N) - CLT(N-1))/(ND(N)-ND(N-1))	    ! Cl input T/day
	    IF((CLT(N-1)-CLT(N)).GT. 0.02*MASS*C(N)) THEN
		DRCL = 0.98 * MASSP * C(N-1)/(MASS * C(N))
!		WRITE(6,1130) YR,MO,DY,DR,DRCL
	    END IF
1130	    FORMAT(' At ',3I4,' Cl dilution query, change from',F6.3,' to',F6.3)

C-- INF is net mass input to lake (KT) 
	    INF = MASSP * (DR - 1.0)	! Input to replace outflow
	    INF = INF + MASS - MASSP	! Input to change total mass	
C-- If lake level near full, or eruptions are causing intermittent outflow, 
C   both terms can apply

C-- Energy balances in TJ; ES = Surface heat loss, EL = change in stored energy
            CALL ES(T(N),W(N),LOSS,EV )
	    E= LOSS + EL(T(N-1),T(N))

C-- NEW FEATURE Solar Incident Radiation Based on yearly guess & month
C-- E is energy required from steam, so is reduced by sun energy
	    E=E - (ND(N) - ND(N-1)) *A* 0.000015 * (1 + 0.5*COS(((MO-1)*3.14)/6.0))


C-- Steam input (in KT) to provide this Energy
	    STEAM = E/(ENTHAL-0.004*T(N))! Energy = Mass * Enthalpy
	    EVAP  = EV			 ! Evaporation loss
	    MELTF = INF + EVAP - STEAM	 ! Conservation of mass

C-- Correction for energy to heat incoming meltwater
C-- FACTOR is ratio: Mass of steam/Mass of meltwater (0 degrees C)
	    FACTOR=T(N)*0.004/(ENTHAL-T(N)*0.004)
	    MELTF = MELTF/(1.0+FACTOR)		! Therefore less meltwater
	    STEAM =STEAM+MELTF*FACTOR		! and more steam
	    E=E+MELTF*T(N)*.004			! Correct energy input also

C	Stable Isotopes calculated now
C	H2 (meltf) now -75, not -57.5 (typo?)	
	    CALL FRACTIONATE(T(N),O18F,H2F)
C	Calculate relative contributions to deltaO18
	    O18M(N) = ( MASSP*O18M(N-1) + STEAM * 10 - MELTF * 10.5 
     &      + EVAP * (8.4 + 1000 * O18F + O18M(N-1)))/MASS
	    H2M(N) = ( MASSP*H2M(N-1) - STEAM * 20 - MELTF * 75.0 
     &      + EVAP * (10.12 + 1000 * H2F + H2M(N-1)))/MASS

C--  Cl in steam by mass   Done if DY = 1, i.e. every month
	    CLTOT = CLTOT + CLT(N) - CLT(N-1) 
	    MGTOT = MGTOT + MGT(N) - MGT(N-1) 
	    EVTOT = EVTOT + EVAP 
	    MTTOT = MTTOT + MELTF 
	    STTOT = STTOT + STEAM
	    TTTOT = TTTOT + T(N)*(ND(N)-ND(N-1))  ! Average Temp Calculation
	    CLDAY = CLDAY + ND(N) - ND(N-1)
	    IF(DY .EQ. 1) THEN
		NCLAV = NCLAV + 1
		NCL(NCLAV) = ND(N)
		TMG(NCLAV) = MGTOT
		TCL(NCLAV) = CLTOT
		TEV(NCLAV) = EVTOT
		TMT(NCLAV) = MTTOT
		STCL(NCLAV)= STTOT
		TAV(NCLAV) = TTTOT/CLDAY
		AVCL(NCLAV)= CLTOT/(STTOT*10)	! Cl (T)/St (KT) is 1/1000
		CLTOT = 0.0			
		MGTOT = 0.0			
		EVTOT = 0.0			
		MTTOT = 0.0			
		STTOT = 0.0
		TTTOT = 0.0
		CLDAY = 0.0
	    END IF	! Chloride % bit
C-- Flows are total amounts/day, 
	    STFL = STEAM/(ND(N)-ND(N-1))	! kT/day
	    PWR  = E/TIMEM			! MW
	    EVFL = EVAP/(ND(N)-ND(N-1))		! kT/day
	    FMELT = MELTF/(ND(N)-ND(N-1))	! kT/day
	    WRITE(6,499) YR,MO,DY,T(N),STFL,PWR,EVFL,
     *	    FMELT,INF,(DR-1.0)*MASSP,MASS-MASSP,FMG(N),FCL(N),O18(N),O18M(N),O18F,H2(N),H2M(N),MASS
  499 	    FORMAT(I4,2I3,F7.1,4F9.1,3F8.1,F7.0,F8.0,5F8.3,F9.3)

C-- Put most flows into common units of Kg/sec or l/sec (difference negligible)
C-- FLOWLINE calculates /day, so allow for this    		
	END DO	! for N

C-- Second Printout
	    WRITE(6,579) 
  579	    FORMAT(//////' YR MO DY TEMP    Outlet    Mg     Cl      DR       MgT   ',
     *	    '    ClT      F Cl    F Mg')
	    DO N=NSTART,NFINISH
	        CALL DATEF(YR,MO,DY,ND(N))
!	        YR=YR-1900
	        WRITE(6,489)YR,MO,DY,T(N),F(N),M(N),C(N),DV(N),MGT(N),CLT(N)
     *	        ,FCL(N),FMG(N)
  489	        FORMAT(I4,2I3,2F7.1,3F8.3,2F10.0,2F9.1)
	    END DO	! for N

C-- Chloride Printout
            
	    WRITE(6,481) 
  481	    FORMAT(////' YR MO DY   CL kT    St kT     %Cl     Ev kT    Mt',
     *      ' kT    T Avge   Mg kT')
 	    DO N=1,NCLAV
		CALL DATEF(YR,MO,DY,NCL(N))
		WRITE(6,479) YR,MO,DY,TCL(N),STCL(N),AVCL(N),TEV(N),
     *      TMT(N),TAV(N),TMG(N)
	    END DO	! For N
  479	    FORMAT(I4,2I3,2F9.1,F8.3,4F9.1)
	    CLOSE(6)
	STOP
	END
	
C =============================================================================
	SUBROUTINE FULLNESS(F)
C	Calculates VOL & A from lake Level NEW VERSION
!  Full is 2529.4 m = Rock Barrier Height
!  Formula has large compensating terms, beware!
!  As in previous versions, VOL is in 000 m3 = kT
C -----------------------------------------------------------------------------
	IMPLICIT NONE
	COMMON /L/ A,VOL
	REAL A,F,VOL
	REAL*8 H,V,V1
	IF(F .LT. 2400.) THEN		! Overflowing Case
	   H = 2529.4
	   VOL = (4.747475*H*H*H-34533.8*H*H+83773360.*H-67772125000.)/1000.
	   A = 193400
	ELSE				! Calculate from absolute level
	   H = F
	   V = 4.747475*H*H*H-34533.8*H*H+83773360.*H-67772125000.
	   H = H + 1.0
	   V1 = 4.747475*H*H*H-34533.8*H*H+83773360.*H-67772125000.
	   A = V1 - V
	   VOL = V/1000.
	END IF
	RETURN
	END

C =============================================================================
	SUBROUTINE FRACTIONATE(T,O18F,H2F)
C	Calculates Fractionation of O18 & H2 as function of temperature
C -----------------------------------------------------------------------------
!   For Bruces Formula, work directly with LnO & LnH
 	IMPLICIT NONE
	REAL*4 T, Tk, O18F, H2F, LnO, LnH
	Tk = T + 273.15
	LnO = (-7.685 + 6712.3/Tk - 1666400./(Tk*Tk) + 350410000/(Tk*Tk*Tk))/1000.
!	O18F = EXP(LnO)
	O18F = LnO
	LnH = (.0000011577*Tk*Tk*Tk - .0016201*Tk*Tk + .79484*Tk - 161.04 + 2999200000./(Tk*Tk*Tk))/1000.
!	H2F = EXP(LnH)
	H2F = LnH
!	PRINT 1000, T, Tk, LnO, LnH, O18F, H2F
 1000	FORMAT(2f7.2,4f14.7)
	RETURN
	END

C =============================================================================
      SUBROUTINE ES(T,W,LOSS,EV )
C     E loss from lake surface in TJ/day
C	Equations apply for nominal surface area of 200000 square metres
C       Since area is only to a small power, this error is negligible
C -----------------------------------------------------------------------------
	IMPLICIT NONE
	COMMON /L/ A,VOL
	REAL*4 T,W,A,VOL,LOSS,EV
	REAL*4 ER,EE,EC,Efree,Eforced
	REAL*4 TK,TL,L
	REAL*4 VP,VP1,VD,Ratio

	L = 500					! Characteristic length of lake			
!  Expressions for H2O properties as function of temperature
! Vapour Pressure Function from CIMO Guide (WMO, 2008)
	VP = 6.112 * exp(17.62*T/(243.12+T))
	VP1= 6.112 * exp(17.62*(T-1.)/(242.12+T)) ! T - 1 for surface T
! Vapour Density from Hyperphysics Site
	VD = .006335 + .0006718*T-.000020887*T*T+.00000073095*T*T*T

!	First term is for radiation, Power(W) = aC(Tw^4 - Ta^4)A 
!	Temperatures absolute, a is emissivity, C Stefans Constant   
	TK = T + 273.15
	TL = 0.9 + 273.15		! 0.9 C is air temperature
	ER = 0.8 * 5.67E-8 * A * (TK*TK*TK*TK -  TL*TL*TL*TL) 
!	ER = 0.8 * 5.67 * A * (TK*TK*TK*TK -  TL*TL*TL*TL) / 100000000

!  Free convection formula from Adams et al(1990) Power (W) = A * factor * delT^1/3 * (es-ea)
!  where factor = 2.3 at 25C and 18% less at 67 C, hence factor = 2.55 - 0.01 * Twr.
!  For both delT and es, we make a 1 C correction, for surface temp below bulk water temp 
!  SVP at average air tmperature 6.5 mBar   

	Efree = A * (2.55-0.01*T) * (T - 1.9) ** (1/3.0) * (VP1 - 6.5)	

!  Forced convection by Satori's Equation  Evaporation (kg/s per m2)= (0.00407 * W**0.8 / L**0.2 - 0.01107/L)(Pw-Pd)/P
!  Latent heat of vapourization about 2400 kJ/kg in range 20 - 60 C, Atmospheric Pressure 750 mBar at Crater Lake

    	Eforced = A * (0.00407 * W**0.8 / L**0.2 - 0.01107/L) * (VP-6.5)/800. * 2400000		! W
	EE = sqrt(Efree*Efree + Eforced*Eforced)

!  The ratio of Heat Loss by Convection to that by Evaporation is rhoCp/L * (Tw - Ta)/(qw - qa)
!  rho is air density .948 kg/m3, Cp Specific Heat of Air 1005 J/kg degC, qw & qa are Sat Vap Density 

	Ratio = .948 * (1005 /2400000.) * (T - 0.9) / (VD - .0022) 
!	print *, W,' ',T, ' ', Efree/1000.,'  ',Eforced/1000.,' ',EE/1000.,' ',Ratio

!  The power calculation is in W. Calculate Energy Loss (TW/day) and
!  evaporative volume loss in kT/day
	EV = 86400 * EE / 2.4E12		! kT/day
	LOSS =(ER + EE*(1+Ratio))*86400*1.0E-12  ! TW/day


C-- 	    Equation values MW (for 200000 sq. metre lake)
  100	CONTINUE
	RETURN
	END


C =============================================================================
	FUNCTION EL(T1,T2)
C	Change in E stored in lake in TJ
C -----------------------------------------------------------------------------
	IMPLICIT NONE
	REAL T1,T2,EL,A,VOL
	COMMON /L/ A,VOL
	EL= (T2-T1) * VOL *.0042		! Interpolate temperature
	RETURN
	END

C =============================================================================
	SUBROUTINE DAYNO(IYR,MO,DY,NO)
C--	Calculates Days since 2000 from Year,Month,Day
C -----------------------------------------------------------------------------
	IMPLICIT NONE
	INTEGER*4 YR,MO,DY
	INTEGER*4 I,IYR, NO
	I=0
	YR=IYR-2000
	IF(MO.LE.2) GOTO 10
	I= MO+3-MO/9
	IF(YR.EQ.(YR/4)*4) I=I-2
   10	CONTINUE
	I=I/2
	NO=DY-1+(MO-1)*31-I+(YR-1)/4 + YR*365
	RETURN
	END

C =============================================================================
	SUBROUTINE DATEF(YR,MO,DY,No)
C--	Calculates Year,Month,Day from days since 2000
C -----------------------------------------------------------------------------
	IMPLICIT NONE
	INTEGER*4 YR,MO,DY,NO,A,B,C,P,Q,R,S
	INTEGER*4 I,IYR
      YR=NO/365
   10 DY=NO-YR*365-(YR-1)/4
      IF(DY.GE.0) GOTO 20
      YR=YR-1
      GOTO 10
   20 MO=DY/31+1
      C=0
      IF(YR.EQ.(YR/4)*4) C=2
   30 A=MO
      B=MO+1
      R=0
      S=0
      IF(A.GT.2) R=(A+3-A/9-C)/2
      P=(A-1)*31-R
      IF(B.GT.2) S=(B+3-B/9-C)/2
      Q=(B-1)*31-S
      IF((DY.GE.P).AND.(DY.LT.Q)) GOTO 40
      MO=MO+1
      GOTO 30
   40 DY=DY-P+1
      YR=YR+2000
      RETURN
      END

