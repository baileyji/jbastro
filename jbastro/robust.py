import numpy as np
import matplotlib.pyplot as plt
def ROB_CHECKFIT(Y, YFIT, EPS, DEL, BFAC=6.0):
    #;+
    #; NAME:
    #;	ROB_CHECKFIT
    #; PURPOSE:
    #;	Used by ROBUST_... routines to determine the quality of a fit and to
    #;	return biweights.
    #; CALLING SEQUENCE:
    #;	status = ROB_CHECKFIT( Y, YFIT, EPS, DEL, SIG, FRACDEV, NGOOD, W, B
    #                          ;				BISQUARE_LIMIT = )
    #; INPUT:
    #;	Y     = the data
    #;	YFIT  = the fit to the data
    #;	EPS   = the "too small" limit
    #;	DEL   = the "close enough" for the fractional median abs. deviations
    #; RETURNS:
    #;	Integer status. if =1, the fit is considered to have converged
    #;
    #; OUTPUTS:
    #;	SIG   = robust standard deviation analog
    #;	FRACDEV = the fractional median absolute deviation of the residuals
    #;	NGOOD   = the number of input point given non-zero weight in the
    #;		calculation
    #;	W     = the bisquare weights of Y
    #;	B     = residuals scaled by sigma
    #;
    #; OPTIONAL INPUT KEYWORD:
    #;	BISQUARE_LIMIT = allows changing the bisquare weight limit from
    #;			default 6.0
    #; PROCEDURES USED:
    #;       ROBUST_SIGMA()
    #; REVISION HISTORY:
    #;	Written, H.T. Freudenreich, HSTX, 1/94
    #;-

    ISTAT = 0
    FRACDEV=None
    NGOOD=None
    W=None
    B=None
    
    DEV = Y-YFIT
    SIG=ROBUST_SIGMA(DEV,ZERO=True)
    #; If the standard deviation = 0 then we're done:
    if SIG < EPS:
        import ipdb;ipdb.set_trace()
        return ISTAT, SIG, FRACDEV, NGOOD, W, B
    
    if DEL > 0.0:
        #; If the fraction std. deviation ~ machine precision, we're done:
        Q_MASK= np.abs(YFIT) > EPS
        COUNT=Q_MASK.sum()
        if COUNT < 3:
            FRACDEV = 0.0
        else:
            # removed /even
            FRACDEV = np.median(np.abs( DEV[Q_MASK]/YFIT[Q_MASK] ))
        if FRACDEV < DEL:
            import ipdb;ipdb.set_trace()
            return ISTAT, SIG, FRACDEV, NGOOD, W, B
    
    ISTAT = 1

    #; Calculate the (bi)weights:
    B = np.abs(DEV)/(BFAC*SIG)
    S_MASK = B > 1.0
    COUNT=S_MASK.sum()
    B[S_MASK] = 1.0
    NGOOD = len(Y)-COUNT
    
    W=(1.0 - B**2)
    W=W/W.sum()

    return ISTAT, SIG, FRACDEV, NGOOD, W, B


def ROBUST_SIGMA(Y, ZERO=False, GOODVEC = False):
    #;
    #;+
    #; NAME:
    #;	ROBUST_SIGMA
    #;
    #; PURPOSE:
    #;	Calculate a resistant estimate of the dispersion of a distribution.
    #; EXPLANATION:
    #;	For an uncontaminated distribution, this is identical to the standard
    #;	deviation.
    #;
    #; CALLING SEQUENCE:
    #;	result = ROBUST_SIGMA( Y, [ /ZERO, GOODVEC = ] )
    #;
    #; INPUT:
    #;	Y = Vector of quantity for which the dispersion is to be calculated
    #;
    #; OPTIONAL INPUT KEYWORD:
    #;	/ZERO - if set, the dispersion is calculated w.r.t. 0.0 rather than the
    #;		central value of the vector. If Y is a vector of residuals, this
    #;		should be set.
    #;
    #; OPTIONAL OUPTUT KEYWORD:
    #;       GOODVEC = Vector of non-trimmed indices of the input vector
    #; OUTPUT:
    #;	ROBUST_SIGMA returns the dispersion. In case of failure, returns
    #;	value of -1.0
    #;
    #; PROCEDURE:
    #;	Use the median absolute deviation as the initial estimate, then weight
    #;	points using Tukey's Biweight. See, for example, "Understanding Robust
    #;	and Exploratory Data Analysis," by Hoaglin, Mosteller and Tukey, John
    #;	Wiley & Sons, 1983, or equation 9 in Beers et al. (1990, AJ, 100, 32)
    #;
    #; REVSION HISTORY:
    #;	H. Freudenreich, STX, 8/90
    #;       Replace MED() call with MEDIAN(/EVEN)  W. Landsman   December 2001
    #;       Don't count NaN values  W.Landsman  June 2010
    #;
    #;-

    
    EPS = 1.0e-20
    Y0=0.0 if ZERO else np.median(Y) #removed /EVEN

    #; First, the median absolute deviation MAD about the median:
    MAD = np.median( np.abs(Y-Y0) )/0.6745 # removed /even

    #; If the MAD=0, try the MEAN absolute deviation:
    if MAD < EPS:
        MAD = ( np.abs(Y-Y0) ).mean()/0.80
    if MAD < EPS:
        import ipdb;ipdb.set_trace()
        return 0.0

    #; Now the biweighted value:
    U   = (Y-Y0)/(6.*MAD)
    UU  = U*U
    Q_MASK=UU < 1.0
    COUNT = Q_MASK.sum()
    if COUNT < 3:
        PRINT('ROBUST_SIGMA: This distribution is TOO WEIRD! Returning -1')
        SIGGMA = -1.
        import ipdb;ipdb.set_trace()
        return SIGGMA
    
    N = np.isfinite(Y).sum()  #In case Y has NaN values          ;
    NUMERATOR = ( (Y[Q_MASK]-Y0)**2 * (1-UU[Q_MASK])**4 ).sum()
    DEN1  = ( (1.-UU[Q_MASK])*(1.-5.*UU[Q_MASK]) ).sum()
    SIGGMA = N*NUMERATOR/(DEN1*(DEN1-1.))
    
    if SIGGMA > 0.0:
        if GOODVEC:
            return np.sqrt(SIGGMA), Q
        else:
            return np.sqrt(SIGGMA)
    else:
        import ipdb;ipdb.set_trace()
        if GOODVEC:
            return 0.0, Q
        else:
            return 0.0


def ROBUST_POLY_FIT(X,Y,NDEG,ITMAX=25):
    #;+
    #; NAME:
    #;	ROBUST_POLY_FIT
    #;
    #; PURPOSE:
    #;	An outlier-resistant polynomial fit.
    #;
    #; CALLING SEQUENCE:
    #;	COEFF = ROBUST_POLY_FIT(X,Y,NDEGREE, [ YFIT,SIG, /DOUBLE, NUMIT=] )
    #;
    #; INPUTS:
    #;	X = Independent variable vector, floating-point or double-precision
    #;	Y = Dependent variable vector
    #;   NDEGREE - integer giving degree of polynomial to fit, maximum = 6
    #; OUTPUTS:
    #;	Function result = coefficient vector, length NDEGREE+1.
    #;	IF COEFF=0.0, NO FIT! If N_ELEMENTS(COEFF) > degree+1, the fit is poor
    #;	(in this case the last element of COEFF=0.)
    #;	Either floating point or double precision.
    #;
    #; OPTIONAL OUTPUT PARAMETERS:
    #;	YFIT = Vector of calculated y's
    #;	SIG  = the "standard deviation" of the residuals
    #;
    #; OPTIONAL INPUT KEYWORD:
    #;       /DOUBLE - If set, then force all computations to double precision.
    #;       NUMIT - Maximum number of iterations to perform, default = 25
    #; RESTRICTIONS:
    #;	Large values of NDEGREE should be avoided. This routine works best
    #;	when the number of points >> NDEGREE.
    #;
    #; PROCEDURE:
    #;	For the initial estimate, the data is sorted by X and broken into
    #;	NDEGREE+2 sets. The X,Y medians of each set are fitted to a polynomial
    #;	 via POLY_FIT.   Bisquare ("Tukey's Biweight") weights are then
    #;	calculated, using a limit  of 6 outlier-resistant standard deviations.
    #;	The fit is repeated iteratively until the robust standard deviation of
    #;	the residuals changes by less than .03xSQRT(.5/(N-1)).
    #;
    #; PROCEDURES CALLED:
    #;        POLY(), POLY_FIT()
    #;       ROB_CHECKFIT()
    #; REVISION HISTORY
    #;	Written, H. Freudenreich, STX, 8/90. Revised 4/91.
    #;	2/94 -- changed convergence criterion
    #;        Added /DOUBLE keyword, remove POLYFITW call  W. Landsman  Jan 2009
    #;-

    EPS   = 1.0E-20
    DEL   = 5.0E-07
    DEGMAX= 6

    BADFIT=0
    LSQFIT=0

    NPTS = len(X)
    MINPTS=NDEG+1
    NEED2 = 1 if (NPTS/4*4) == NPTS else 0
    N3 = 3*NPTS/4
    N1 = NPTS/4

    #; If convenient, move X and Y to their centers of gravity:
    if NDEG < DEGMAX:
        X0=X.sum()/NPTS
        Y0=Y.sum()/NPTS
        U=X-X0
        V=Y-Y0
    else:
        U=X.copy()
        V=Y.copy()

    #; The initial estimate.

    #; Choose an odd number of segments:
    NUM_SEG = NDEG+2
    if (NUM_SEG/2*2) == NUM_SEG:
        NUM_SEG =NUM_SEG+1
    MIN_PTS = NUM_SEG*3

    if NPTS < 10000: #MIN_PTS THEN BEGIN
        #;  Settle for least-squares:
        LSQFIT = 1
        CC = np.polyfit( U, V, NDEG)
        YFIT=np.poly1d(CC)(U)
    else:
        #;  Break up the data into segments:
        LSQFIT = 0
        Q = U.argsort()
        U=U[Q]
        V = V[Q]
        N_PER_SEG = np.zeros(NUM_SEG) + NPTS/NUM_SEG

        #;  Put the leftover points in the middle segment:
        N_LEFT = NPTS - N_PER_SEG[0]*NUM_SEG
        N_PER_SEG[NUM_SEG/2] = N_PER_SEG[NUM_SEG/2] + N_LEFT
        R = np.zeros(NUM_SEG)
        S = np.zeros(NUM_SEG)
        R[0]=np.median( U[0:N_PER_SEG[0]]) #removed /even
        S[0]=np.median( V[0:N_PER_SEG[0]]) #removed /even
        I2 = N_PER_SEG[0]-1
        for I in range(1,NUM_SEG):
            I1 = I2 + 1
            I2 = I1 + N_PER_SEG[I]
            R[I] = np.median( U[I1:I2]) #removed /even
            S[I] = np.median( V[I1:I2]) #removed /even
        
        #;  Now fit:
        CC = np.polyfit( R,S, NDEG)
        YFIT = np.poly1d(CC)(U)


    ISTAT,SIG,FRACDEV,NGOOD,W,S = ROB_CHECKFIT(V,YFIT,EPS,DEL)

    if ISTAT != 0 and NGOOD < MINPTS:
        if LSQFIT == 0:
            #;  Try a least-squares:
            CC = np.polyfit( U, V, NDEG)
            YFIT = np.poly1d(CC)(U)
            ISTAT,SIG,FRACDEV,NGOOD,W,S = ROB_CHECKFIT(V,YFIT,EPS,DEL)
            #NGOOD = NPTS-COUNT #This line is a bug in the original
        if ISTAT !=0 and NGOOD < MINPTS:
            import ipdb;ipdb.set_trace()
            raise ValueError('ROBUST_POLY_FIT: No Fit Possible!')
            return 0.0,None,SIG
    
    if ISTAT != 0:
        #; Now iterate until the solution converges:
        CLOSE_ENOUGH = max(.03*np.sqrt(.5/(NPTS-1)), DEL)
        DIFF= 1.0e10
        SIG_1= min((100.0*SIG), 1.0e20)
        NIT = 0
        while ( (DIFF > CLOSE_ENOUGH) and (NIT < ITMAX) ):
            NIT=NIT+1
            SIG_2=SIG_1
            SIG_1=SIG
            #; We use the "obsolete" POLYFITW routine because it allows
            #   us to input weights
            #; rather than measure errors
            g_mask=W > 0
            #;Throw out points with zero weight
            if g_mask.sum() < len(W):
                U = U[g_mask]
                V = V[g_mask]
                W = W[g_mask]
            CC = np.polyfit( U, V, NDEG, w = 1.0/W**2)
            YFIT = np.poly1d(CC)(U)
            ISTAT,SIG,FRACDEV,NGOOD,W,S = ROB_CHECKFIT(V,YFIT,EPS,DEL)
            if ISTAT==0:
                break
            if NGOOD < MINPTS:
                print('ROBUST_POLY_FIT: Questionable Fit!')
                BADFIT=1
                break
            DIFF = min((np.abs(SIG_1-SIG)/SIG) , (np.abs(SIG_2-SIG)/SIG) )

        #;IF NIT GE ITMAX THEN PRINT,'ROBUST_POLY_FIT: Did not converge in',ITMAX,$
        #;' iterations!'

    if NDEG < DEGMAX:
        if NDEG == 1:
            CC[0] = CC[0]-CC[1]*X0 + Y0
        elif NDEG==2:
            CC[0] = CC[0]-CC[1]*X0+CC[2]*X0**2 + Y0
            CC[1] = CC[1]-2.*CC[2]*X0
        elif NDEG==3:
            CC[0] = CC[0]-CC[1]*X0+CC[2]*X0**2-CC[3]*X0**3 + Y0
            CC[1] = CC[1]-2.*CC[2]*X0+3.*CC[3]*X0**2
            CC[2] = CC[2]-3.*CC[3]*X0
        elif NDEG==4:
            CC[0] = CC[0]-   CC[1]*X0+CC[2]*X0**2-CC[3]*X0**3+CC[4]*X0**4+ Y0
            CC[1] = CC[1]-2.*CC[2]*X0+3.*CC[3]*X0**2-4.*CC[4]*X0**3
            CC[2] = CC[2]-3.*CC[3]*X0+6.*CC[4]*X0**2
            CC[3] = CC[3]-4.*CC[4]*X0
        elif NDEG==5:
            CC[0] = (CC[0]-CC[1]*X0+CC[2]*X0**2-
                     CC[3]*X0**3+CC[4]*X0**4-CC[5]*X0**5 + Y0)
            CC[1] = (CC[1]-2.*CC[2]*X0+ 3.*CC[3]*X0**2-
                     4.*CC[4]*X0**3+5.*CC[5]*X0**4)
            CC[2] = CC[2]-3.*CC[3]*X0+ 6.*CC[4]*X0**2-10.*CC[5]*X0**3
            CC[3] = CC[3]-4.*CC[4]*X0+10.*CC[5]*X0**2
            CC[4] = CC[4]-5.*CC[5]*X0

    #; Calculate the fit at points X:
    YFIT=np.poly1d(CC)(X)

    if BADFIT== 1:
        CC=np.concatenate(np.array([0.]),CC)
        YFIT=None

    return CC, YFIT,SIG

