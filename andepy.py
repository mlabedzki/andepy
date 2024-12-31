import numpy as np
import numba as nb

@nb.jit(nopython=True)
def is_invertible_jit(A):
    out = A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]
    return out

@nb.jit(nopython=True)
def ols_coef_jit(y,x):
    xxinv=np.dot(x.T,x)
    if is_invertible_jit(xxinv):
        xxinv = np.linalg.inv(xxinv)
        xy=np.dot(x.T,y)
        coef=np.dot(xxinv,xy)
    return coef

@nb.jit(nopython=True)
def nanmean2d_jit(data, axis=0): # mean function for 2d matrix with axis parameter and omitting of nans
    n=data.shape[0] #==len(data)
    k=data.shape[1] #==len(x[0])
    ydata=data.copy()
    counts=np.zeros(k)
    for j in range(k):
        ydata_j = ydata[:,j]
        mask = np.isnan(ydata_j)
        counts[j] = mask.sum()
        ydata_j[mask] = 0
        ydata[:,j]=ydata_j
    out = np.sum(ydata,axis)/(n-counts)
    return out

@nb.jit(nopython=True)
def sdratio_jit(sd,sd2): #computes sd's ratio
    ##below is alternate version but it is a little slower
    #increment = 0.000000001
    #out=np.sign(sd-sd2)*(1-(np.minimum(sd,sd2)+increment)/(np.maximum(sd,sd2)+increment))
    ##or
    #out=np.sign(sd-sd2)*(1-np.minimum((sd+increment)/(sd2+increment),(sd2+increment)/(sd+increment)))
    n=len(sd)
    out=np.zeros(n)
    for i in range(n):
        if(sd[i]>sd2[i]):
            out[i] = 1-sd2[i]/sd[i]
        elif(sd[i]<sd2[i]):
            out[i] = sd[i]/sd2[i]-1
    return out

@nb.jit(nopython=True)
def nancov_jit(mat): #computes cov with omitting nan's, the mean of each col is assumed to be 0
    data=mat.copy()
    n=data.shape[0] #==len(data)
    k=data.shape[1] #==len(data[0])
    nans=np.ones((n,k))
    for i in range(n):
        for j in range(k):    
            if np.isnan(data[i, j]):
                nans[i, j] = 0
                data[i, j] = 0
    out = np.dot(data.T,data)/(np.dot(nans.T,nans))
    for i in range(k):
        for j in range(k):    
            if ((np.isnan(out[i, j])) or (out[i,j]==np.inf) or (out[i,j]==-np.inf)):
                out[i, j] = 0
    return out

@nb.jit(nopython=True)
def nancov2_jit(mat): #computes cov with omitting nan's, the mean of each col is assumed to be 0
    data=mat.copy()
    n=data.shape[0] #==len(data)
    k=data.shape[1] #==len(data[0])
    nans=np.ones((n,k))
    for i in range(n):
        for j in range(k):    
            if np.isnan(data[i, j]):
                nans[i, j] = 0
                data[i, j] = 0
    counts = np.dot(nans.T,nans)
    out = np.dot(data.T,data)/counts
    for i in range(k):
        for j in range(k):    
            if ((np.isnan(out[i, j])) or (out[i,j]==np.inf) or (out[i,j]==-np.inf)):
                out[i, j] = 0
    return out,counts

@nb.jit(nopython=True)
def covex1r_jit(dmat,nanhelper,t=-1,j=-1): #computes cov for matrix where some row is deleted in a special mode
    data=dmat.copy() #we deal with return matrix in pandas sense, first row is nan
    n=data.shape[0] #==len(data)
    k=data.shape[1] #==len(data[0])
    # here we will delete one row and add it to nearby row - to get main effect
    if t>-1: #this is for value score mode
        if(t<len(data)):            
            zz = data.copy() #we create another object to avoid compiler confusion
            if(t+1<len(data)):
                for i in range(0,k):
                    if(~np.isnan(zz[t,i])):
                        shift = nanhelper[t,i] #we will add return we are going to remove to next nonnan return, nanhelper is keeping how many nans are skipped this way
                        zz[t+1+shift,i] = (zz[t+1+shift,i]+zz[t,i])/np.sqrt(2+shift)                            
            zz[t,:] = np.nan                
            data = zz
    if j>-1: #this is for adjfactor score mode
        if(j<len(data)):
            rowmask = np.array([x!=j for x in np.arange(0,n)])
            data = data[rowmask,:] #introduce empty return for whole row (day) by deleting it
    cv = nancov_jit(data)
    return cv

@nb.jit(nopython=True)
def mat_prep_step1_jit(mat): #helper function preparing returns matrix, it keeps dimensions unchanged
    data = mat.copy()
    n = data.shape[0] #==len(data)
    k = data.shape[1] #==len(data[0])
    for i in np.arange(0,k):
        data_i = np.vstack((data[:,i],np.arange(0,n))).T #add extra indexation
        idx = np.arange(0,n)
        ddata_i = data_i[(~np.isnan(data_i[:,0])),:]
        idx = idx[(~np.isnan(data_i[:,0]))]
        ddata_i = np.diff(ddata_i.copy().T,1).T #we transpose back and forth to enable diff by rows instead of columns
        idx = idx[1:]
        # here we will bring back rows with NA's
        data_i[idx,:] = ddata_i
        data_i[0,0] = np.nan
        data[:,i] = (data_i[:,0] / np.sqrt(data_i[:,1])) #divide returns by sqrt(overlap), where overlap = extra_indexation.diff()
    return data

@nb.jit(nopython=True)
def mat_prep_step2_jit(mat): #helper function filling some nans in returns matrix   
    data = mat.copy()
    n = data.shape[0] #==len(data)
    k = data.shape[1] #==len(data[0])    
    tt = np.isnan(mat)
    ttrowsum = tt.sum(1)
    for day_i in np.arange(1,n-1):
        if (ttrowsum[day_i]!=k):
            for i in np.arange(0,k): # below filling corrections allow to analyse matrix with nan in 'chess board pattern'
                if tt[day_i][i] and ttrowsum[day_i-1]+ttrowsum[day_i+1]>0:
                    data[day_i][i] = data[day_i+1][i]/np.sqrt(2)
                    data[day_i+1][i] = data[day_i+1][i]/np.sqrt(2)
    return data
                    
@nb.jit(nopython=True)
def mat_prep_step3_jit(data): #helper function preparing helper matrix that counts nans
    n = data.shape[0] #==len(data)
    k = data.shape[1] #==len(data[0]) 
    nanhelper = np.zeros((n,k),dtype=np.int64)
    tt = np.isnan(data)
    for i in np.arange(0,k):
        counter = 0
        for day_i in np.arange(n-1,-1,-1):
            if tt[day_i][i]:
                counter += 1
            else:
                if(day_i + 1 + counter < n):
                    nanhelper[day_i][i] = counter
                else:
                    nanhelper[day_i][i] = counter-1 # we protect here against a situation where nanhelper would lead us to n+1 row that does not exist                    
                counter = 0
    return nanhelper

@nb.jit(nopython=True)
def anomalies_main(vdata,logreturn=True,is_returns=False): #main anomaly detector function
    '''
    Parameters
    ----------
    vdata : np array with time series data to evaluate, time series should be stored vertical
    logreturn : return type, True for logreturn (default), False for abs return
    is_returns : set True if input is already as returns, set False (default) if input is in levels

    Returns a touple with two elements where:
    - out stores np array with Value scores
    - outAF stores np array with Adjustment Factor scores
    '''
    
    n = vdata.shape[0] #==len(vdata)
    k = vdata.shape[1] #==len(vdata[0])
    if logreturn:
        data = np.log(vdata.copy())
    else:
        data = vdata.copy()
    if not is_returns:
        data = mat_prep_step1_jit(data)
    data = mat_prep_step2_jit(data)
    nanhelper = mat_prep_step3_jit(data)
    #below we also create dataAF, which is for AdjFactor analysis, here we fill even more singleton NA's
    dataAF = data.copy()
    tt = np.isnan(dataAF)
    ttrowsum = tt.sum(1)    
    for day_i in np.arange(1,n-1):
        if (ttrowsum[day_i]!=k):
            for i in np.arange(0,k): # below filling corrections allow to analyse adj factor as it should be for singleton NA's
                if tt[day_i][i]:# "and ttrowsum[day_i-1]+ttrowsum[day_i+1]>0:" part is not required this time as for ADJ FACTOR we need to fill all singleton NA's for returns
                    dataAF[day_i][i] = dataAF[day_i+1][i]/np.sqrt(2)
                    dataAF[day_i+1][i] = dataAF[day_i+1][i]/np.sqrt(2)    
    cv = covex1r_jit(data,nanhelper)
    sd = np.sqrt(np.diag(cv))
    sdb=sd.copy()
    sdb[sdb==0]=1 # to decrease number of nan's in corr matrix and sd2/sd ratios, alternative to such operation with all sd's we could later call: nansum_jit(cr2**2 - cr**2,1) for value and AF, plus out[day_i][np.isnan(out[day_i,:])] = 0 for value and AF, to additionaly make out=0 for all cases where it is nan due to sd=0
    cr = cv/(np.outer(sdb,sdb))
    for i in np.arange(0,k): # to get rid of cov(x,x)/sd(x)^2 where we had 0/0, all others are guaranted to be 1
        cr[i,i] = 1
    out = np.empty((n,k,))
    out[:] = np.nan
    outAF = np.empty((n,k,))
    outAF[:] = np.nan
    for day_i in np.arange(0,n):
        #ADJ FACTOR score part
        cv2AF = covex1r_jit(dataAF,nanhelper,-1,day_i+1)
        sd2AF = np.sqrt(np.diag(cv2AF))
        sd2AFb=sd2AF.copy()
        sd2AFb[sd2AFb==0]=1
        cr2AF = cv2AF/(np.outer(sd2AFb,sd2AFb))
        for i in np.arange(0,k):
            cr2AF[i,i] = 1
        outAF[day_i][:] = ( (cr2AF**2 - cr**2).sum(1) + sdratio_jit(sd,sd2AF) ) * 100/k
        #VALUE score part
        mask = np.isnan(vdata[day_i,:])
        data_i_nan = data.copy()
        data_i_nan[:,mask] = 0
        #we run covex1r_jit(data_i_nan,nanhelper,day_i,-1) with cov(data_i_nan) being equivalent to cov() on data matrix where columns with NA on this day were deleted, then we force result into bigger matrix and omitted columns receive zeros
        cv2 = covex1r_jit(data_i_nan,nanhelper,day_i,-1)      
        sd2 = np.sqrt(np.diag(cv2))
        sd2[mask] = 1 # we set as 1 as we do not need it and due to 0's previously it is nan's, also 1 allows to divide without error
        sd2b=sd2.copy()
        sd2b[sd2b==0]=1    
        cr2 = cv2/(np.outer(sd2b,sd2b))
        for i in np.arange(0,k):
            cr2[i,i] = 1
        cr_masked = cr.copy()
        cr_masked[:,mask] = 0 # we set whole columns to zero, later we will sum differences in rows and it is on purpose
        for i in np.arange(0,k):
            cr_masked[i,i] = 1        
        out[day_i][:] = ( (cr2**2 - cr_masked**2).sum(1) + sdratio_jit(sd,sd2) ) * 100/k
        out[day_i][mask] = 0 # mask to make out=0 for cases where vdata=nan
    for day_i in np.arange(n-1,1,-1):#we move assesment towards first nonnan before series of nan, we start from end so we backpropoagate
        for ts_i in np.arange(0,k):
            if np.isnan(vdata[day_i][ts_i]):
                outAF[day_i-1][ts_i] = outAF[day_i][ts_i]     
                outAF[day_i][ts_i] = 0
    out[0,:] = outAF[0,:] #that is the rule
    return out,outAF
   
def anomalies_sort(mat): #anomalies sorting for 1 type of score 
    XX,YY = np.meshgrid(np.arange(mat.shape[1]),np.arange(mat.shape[0]))
    mat_flat = np.vstack((YY.ravel(),XX.ravel(),mat.ravel())).T
    mat_flat = mat_flat[mat_flat[:, 2].argsort()][::-1]
    dt = {'names':['Date','MKF','Score'], 'formats':[int, int, float]}
    table = np.empty(len(mat_flat), dtype=dt)
    table[dt['names'][0]] = mat_flat[:,0]
    table[dt['names'][1]] = mat_flat[:,1]
    table[dt['names'][2]] = mat_flat[:,2]
    return table

def anomalies_sort2d(mat,matAF): #anomalies sorting for 2 types of scores, using max of both as sorting column
    XX,YY = np.meshgrid(np.arange(mat.shape[1]),np.arange(mat.shape[0]))
    mat_flat = np.vstack((YY.ravel(),XX.ravel(),np.maximum(mat.ravel(),matAF.ravel()),mat.ravel(),matAF.ravel(),)).T
    mat_flat = mat_flat[mat_flat[:, 2].argsort()][::-1]
    dt = {'names':['Date','MKF','Max Score','Value Score','AdjFactor Score','AdjFactor Type'], 'formats':[int, int, float, float, float, bool]}
    table = np.empty(len(mat_flat), dtype=dt)
    table[dt['names'][0]] = mat_flat[:,0]
    table[dt['names'][1]] = mat_flat[:,1]
    table[dt['names'][2]] = mat_flat[:,2]
    table[dt['names'][3]] = mat_flat[:,3]
    table[dt['names'][4]] = mat_flat[:,4]
    table[dt['names'][5]] = table['Value Score']<table['AdjFactor Score']
    return table


#########################
### POINT SOLVER PART ###
#########################

#source: https://github.com/limix/brent-search/blob/master/brent_search/_brent.py
inf = float("inf")
_eps = 1.4902e-08
_golden = 0.381966011250105097
def brent(f, a=-inf, b=+inf, x0=None, f0=None, rtol=_eps, atol=_eps, maxiter=500):
    """Seeks a minimum of a function via Brent's method.
    
    Given a function ``f`` with a minimum in the interval ``a <= b``, seeks a local
    minimum using a combination of golden section search and successive parabolic
    interpolation.
    
    Let ``tol = rtol * abs(x0) + atol``, where ``x0`` is the best guess found so far.
    It converges if evaluating a next guess would imply evaluating ``f`` at a point that
    is closer than ``tol`` to a previously evaluated one or if the number of iterations
    reaches ``maxiter``.
    
    Parameters
    ----------
    f : object
        Objective function to be minimized.
    a : float, optional
        Interval's lower limit. Defaults to ``-inf``.
    b : float, optional
        Interval's upper limit. Defaults to ``+inf``.
    x0 : float, optional
        Initial guess. Defaults to ``None``, which implies that::
            
            x0 = a + 0.382 * (b - a)
            f0 = f(x0)
    
    f0 : float, optional
        Function evaluation at ``x0``.
    rtol : float
        Relative tolerance. Defaults to ``1.4902e-08``.
    atol : float
        Absolute tolerance. Defaults to ``1.4902e-08``.
    maxiter : int
        Maximum number of iterations.
    
    
    Returns
    -------
    float
        Best guess ``x`` for the minimum of ``f``.
    float
        Value ``f(x)``.
    int
        Number of iterations performed.
    
    References
    ----------
    - http://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.c
    - Numerical Recipes 3rd Edition: The Art of Scientific Computing
    - https://en.wikipedia.org/wiki/Brent%27s_method
    """
    # a, b: interval within the minimum should lie
    #       no function evaluation will be requested
    #       outside that range.
    # x0: least function value found so far (or the most recent one in
    #                                            case of a tie)
    # x1: second least function value
    # x2: previous value of x1
    # (x0, x1, x2): Memory triple, updated at the end of each interation.
    # u : point at which the function was evaluated most recently.
    # m : midpoint between the current interval (a, b).
    # d : step size and direction.
    # e : memorizes the step size (and direction) taken two iterations ago
    #      and it is used to (definitively) fall-back to golden-section steps
    #      when its value is too small (indicating that the polynomial fitting
    #      is not helping to speedup the convergence.)
    #
    #
    # References: Numerical Recipes: The Art of Scientific Computing
    # http://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.c
    
    if a > b:
        raise ValueError("'a' must be equal or smaller than 'b'")
    
    if x0 is None:
        x0 = a + _golden * (b - a)
        f0 = f(x0)
    else:
        if not (a <= x0 <= b):
            raise RuntimeError("'x0' didn't fall in-between 'a' and 'b'")
    
    x1 = x0
    x2 = x1
    niters = -1
    d = 0.0
    e = 0.0
    f1 = f0
    f2 = f1
    
    for niters in range(maxiter):
        
        m = 0.5 * (a + b)
        tol = rtol * abs(x0) + atol
        tol2 = 2.0 * tol
        
        # Check the stopping criterion.
        if abs(x0 - m) <= tol2 - 0.5 * (b - a):
            break
        
        r = 0.0
        q = r
        p = q
        
        # "To be acceptable, the parabolic step must (i) fall within the
        # bounding interval (a, b), and (ii) imply a movement from the best
        # current value x0 that is less than half the movement of the step
        # before last."
        #   - Numerical Recipes 3rd Edition: The Art of Scientific Computing.
        
        if tol < abs(e):
            # Compute the polynomial of the least degree (Lagrange polynomial)
            # that goes through (x0, f0), (x1, f1), (x2, f2).
            r = (x0 - x1) * (f0 - f2)
            q = (x0 - x2) * (f0 - f1)
            p = (x0 - x2) * q - (x0 - x1) * r
            q = 2.0 * (q - r)
            if 0.0 < q:
                p = -p
            q = abs(q)
            r = e
            e = d
        
        if abs(p) < abs(0.5 * q * r) and q * (a - x0) < p and p < q * (b - x0):
            # Take the polynomial interpolation step.
            d = p / q
            u = x0 + d
            
            # Function must not be evaluated too close to a or b.
            if (u - a) < tol2 or (b - u) < tol2:
                if x0 < m:
                    d = tol
                else:
                    d = -tol
        else:
            # Take the golden-section step.
            if x0 < m:
                e = b - x0
            else:
                e = a - x0
            d = _golden * e
        
        # Function must not be evaluated too close to x0.
        if tol <= abs(d):
            u = x0 + d
        elif 0.0 < d:
            u = x0 + tol
        else:
            u = x0 - tol
        
        # Notice that we have u \in [a+tol, x0-tol] or
        #                     u \in [x0+tol, b-tol],
        # (if one ignores rounding errors.)
        fu = f(u)
        
        # Housekeeping.
        
        # Is the most recently evaluated point better (or equal) than the
        # best so far?
        if fu <= f0:
            
            # Decrease interval size.
            if u < x0:
                if b != x0:
                    b = x0
            else:
                if a != x0:
                    a = x0
            
            # Shift: drop the previous third best point out and
            # include the newest point (found to be the best so far).
            x2 = x1
            f2 = f1
            x1 = x0
            f1 = f0
            x0 = u
            f0 = fu
        
        else:
            # Decrease interval size.
            if u < x0:
                if a != u:
                    a = u
            else:
                if b != u:
                    b = u
            
            # Is the most recently evaluated point at better (or equal)
            # than the second best one?
            if fu <= f1 or x1 == x0:
                # Insert u between (rank-wise) x0 and x1 in the triple
                # (x0, x1, x2).
                x2 = x1
                f2 = f1
                x1 = u
                f1 = fu
            elif fu <= f2 or x2 == x0 or x2 == x1:
                # Insert u in the last position of the triple (x0, x1, x2).
                x2 = u
                f2 = fu
    
    return x0, f0, niters + 1

@nb.jit(nopython=True)
def point_solver_prepare(odata,day_i,mkf_j): # point solver function preparing data before optimization
    data = odata.copy()
    k = data.shape[1] #==len(data[0])
    data[day_i,mkf_j] = np.nan
    tmp = data[1:,:]
    cv,counts = nancov2_jit(tmp)
    sd = np.sqrt(np.diag(cv))
    sdb=sd.copy()
    sdb[sdb==0]=1 # to decrease number of nan's in corr matrix and sd2/sd ratios, alternative to such operation with all sd's we could later call: nansum_jit(cr2**2 - cr**2,1) for value and AF, plus out[day_i][np.isnan(out[day_i,:])] = 0 for value and AF, to additionaly make out=0 for all cases where it is nan due to sd=0
    cr = cv/(np.outer(sdb,sdb))
    for i in np.arange(0,k): # to get rid of cov(x,x)/sd(x)^2 where we had 0/0, all others are guaranted to be 1
        cr[i,i] = 1
    data_i_new = data[day_i,:].copy()
       
    return (data_i_new,mkf_j,counts[:,mkf_j],cv[:,mkf_j],cr[:,mkf_j],sd)

@nb.jit(nopython=True)
def point_solver_func_inner(val,data_i_new,mkf_j,counts_j,cv_j,cr_j,sd): #point solver function that is minimized
    #k=len(sd)
    sd_new = sd.copy()
    sd_new[mkf_j] = np.sqrt((sd[mkf_j]**2*counts_j[mkf_j] + val**2)/(counts_j[mkf_j]+1))
    data_i_new[mkf_j] = val   
    cr_new = ((cv_j*counts_j + data_i_new*val)/(counts_j+1))/(sd_new*sd_new[mkf_j])    
    score = np.abs(cr_j**2 - cr_new**2).sum() 
    score = score + np.log(sd[mkf_j]/sd_new[mkf_j])
    #some other formulas that were considered:
    #score = np.abs(cr_j**2 - cr_new*cr_j).sum() 
    #cr_new = ((cv_j*counts_j + data_i_new*val)/(counts_j+1))/(sd*sd_new[mkf_j])    
    #score = score + np.log(np.minimum(sd[mkf_j],sd_new[mkf_j])/(np.maximum(sd[mkf_j],sd_new[mkf_j])))
    #score = score + np.sign(sd[mkf_j]-sd_new[mkf_j])*(1-(np.minimum(sd[mkf_j],sd_new[mkf_j]))/(np.maximum(sd[mkf_j],sd_new[mkf_j])))
    #score = score + (sd[mkf_j]/sd_new[mkf_j])-1
    #score = ((cr_j**2 - cr_new**2).sum() + np.sign(sd[mkf_j]-sd_new[mkf_j])*(1-(np.minimum(sd[mkf_j],sd_new[mkf_j]))/(np.maximum(sd[mkf_j],sd_new[mkf_j]))) ) * 100/k
    #cv_new = ((cv_j*counts_j + data_i_new*val)/(counts_j+1))
    #score = (np.abs(cv_j - cv_new)).sum()
    return np.abs(score)

def point_solver(vdata,day_i,mkf_j,logreturn=True,is_returns=False,x0=None): #point solver main function
    '''
    Parameters
    ----------
    vdata : np array with time series data to evaluate, time series should be stored vertical
    day_i : integer describing day in the vdata array
    mkf_j : integer describing mkf in the vdata array
    logreturn : return type, True for logreturn, False for abs return
    is_returns : set True if input is already as returns, set False (default) if input is in levels
    x0 : float number that will be used as starting point for expected return search, by default 0
    
    Returns the expected return on considered day
    '''
    k = vdata.shape[1] #==len(vdata[0])
    if logreturn:
        data = np.log(vdata.copy())
    else:
        data = vdata.copy()
    #data[day_i,mkf_j] = np.nan
    if not is_returns:
        data = mat_prep_step1_jit(data)
        data = mat_prep_step2_jit(data)
        
    data_i_new,mkf_j,counts_j,cv_j,cr_j,sd = point_solver_prepare(data,day_i,mkf_j)
    def f(x):
        return point_solver_func_inner(x, data_i_new,mkf_j,counts_j,cv_j,cr_j,sd)

    if x0 is None:    
        x0 = data[day_i,mkf_j]
        if np.isnan(x0):
            regdata=data[~np.isnan(data).any(axis=1), :]
            x = np.delete(regdata, mkf_j, axis=1)
            x = [np.ones(x.shape[0]), x]
            x = np.column_stack(x)
            y = regdata[:,mkf_j].copy()
            params = ols_coef_jit(y,x)
            #alternatively it could be done by:
            #import statsmodels.api as sm
            #params = (sm.OLS(endog=y, exog=x).fit()).params
            try:
                x0 = (params*np.append(1, data[day_i,np.arange(k)!=mkf_j])).sum()
            except:
                x0 = np.nan
        if np.isnan(x0):
            avg = nanmean2d_jit(data, 0)
            x0 = avg[mkf_j]
    if sd[mkf_j]==0:
        out = x0
    else:
        out = brent(f, a=min(x0,-5*sd[mkf_j]), b=max(x0,5*sd[mkf_j]), x0=x0, f0=f(x0), rtol=_eps, atol=_eps, maxiter=500)[0]
    # as alternative we could use:
    # import scipy.optimize as spo
    # out = spo.minimize_scalar(lambda x: point_solver_func_inner(x,data_i_new,mkf_j,counts_j,cv_j,cr_j,sd), method='Brent')['x']
    return out
       
def exp_score(vdata,logreturn=True,is_returns=False,is_reg=True): #main exp score procedure
    '''
    Parameters
    ----------
    vdata : np array with time series data to evaluate, time series should be stored vertical
    logreturn : return type, True for logreturn, False for abs return
    is_returns : set True if input is already as returns, set False (default) if input is in levels
    is_reg : set True (default) if search for expected point should start in lin regression fitted value, set False if it should start in original point
    
    Returns exp scores for all days in the matrix, exp score is standardized deviation of the return from expected return
    '''

    n = vdata.shape[0] #==len(vdata)
    k = vdata.shape[1] #==len(vdata[0])
    if logreturn:
        data = np.log(vdata.copy())
    else:
        data = vdata.copy()
    if not is_returns:
        data = mat_prep_step1_jit(data)
    data = mat_prep_step2_jit(data)
        
    avg = nanmean2d_jit(data, 0)   
    sd = np.sqrt(nanmean2d_jit(data**2, 0) - avg**2)
    sdinv = 1/sd
    sdinv[sdinv==np.inf]=0
    
    dev_ret = np.zeros((n,k))
    for j in range(0,k):
        print(str(j)+"/"+str(k))
        if is_reg:
            regdata=data[~np.isnan(data).any(axis=1), :]
            x = np.delete(regdata, j, axis=1)
            x = [np.ones(x.shape[0]), x]
            x = np.column_stack(x)
            y=regdata[:,j].copy()
            params = ols_coef_jit(y,x)
            #alternatively it could be done by:
            #import statsmodels.api as sm
            #params = (sm.OLS(endog=y, exog=x).fit()).params
        for i in range(1,n):
            if not np.isnan(data[i,j]):
                if is_reg:
                    try:
                        x0=(params*np.append(1, data[i,np.arange(k)!=j])).sum()               
                    except:
                        x0 = np.nan                    
                    if np.isnan(x0):
                        x0=avg[j]
                    #data_i_new,mkf_j,counts_j,cv_j,cr_j,sd_loc = point_solver_prepare(data,i,j)
                    #f0 = point_solver_func_inner(x0, data_i_new,mkf_j,counts_j,cv_j,cr_j,sd_loc)
                    #fbasis = point_solver_func_inner(data[i,j], data_i_new,mkf_j,counts_j,cv_j,cr_j,sd_loc)
                    #if f0>fbasis:
                    #    x0=data[i,j]
                else:
                    x0=data[i,j]
                dev_ret[i,j] = data[i,j] - point_solver(data,i,j,logreturn=False,is_returns=True,x0=x0)
    out = abs(dev_ret)*sdinv
    out[0,:] = 0
    return out

def oos_exp_score(vdata,logreturn=True,is_returns=False,is_reg=False,window=30,k_limit=None): #main oos exp score procedure
    '''
    Parameters
    ----------
    vdata : np array with time series data to evaluate, time series should be stored vertical
    logreturn : return type, True for logreturn, False for abs return
    is_returns : set True if input is already as returns, set False (default) if input is in levels
    is_reg : set True (default) if search for expected point should start in lin regression fitted value, set False if it should start in original point
    
    Returns out of sample exp scores for all days after window period in the matrix, exp score is standardized deviation of the return from expected return
    '''
#vdata=mydata.values
    n = vdata.shape[0] #==len(vdata)
    k = vdata.shape[1] #==len(vdata[0])
    if logreturn:
        data = np.log(vdata.copy())
    else:
        data = vdata.copy()
    if not is_returns:
        data = mat_prep_step1_jit(data)
    data = mat_prep_step2_jit(data)

    exp_ret = np.zeros((n,k))
    for j in range(0,k):
        print(str(j)+"/"+str(k))
        for i in range(window,n):
            #print(str(i)+"/"+str(n))
            locdata=data[(i-window):(i+1),:]
            avg = nanmean2d_jit(locdata, 0)   
            sd = np.sqrt(nanmean2d_jit(locdata**2, 0) - avg**2)
            sdinv = 1/sd
            sdinv[sdinv==np.inf]=0            
            if is_reg:
                regdata=locdata[~np.isnan(locdata).any(axis=1), :]
                x = np.delete(regdata, j, axis=1)
                x = [np.ones(x.shape[0]), x]
                x = np.column_stack(x)
                y=regdata[:,j].copy()
                params = ols_coef_jit(y,x)
                #alternatively it could be done by:
                #import statsmodels.api as sm
                #params = (sm.OLS(endog=y, exog=x).fit()).params            
            if not np.isnan(data[i,j]):
                if is_reg:
                    try:
                        x0=(params*np.append(1, data[i,np.arange(k)!=j])).sum()               
                    except:
                        x0 = np.nan                    
                    if np.isnan(x0):
                        x0=data[i,j]
                    #data_i_new,mkf_j,counts_j,cv_j,cr_j,sd_loc = point_solver_prepare(data,i,j)
                    #f0 = point_solver_func_inner(x0, data_i_new,mkf_j,counts_j,cv_j,cr_j,sd_loc)
                    #fbasis = point_solver_func_inner(data[i,j], data_i_new,mkf_j,counts_j,cv_j,cr_j,sd_loc)
                    #if f0>fbasis:
                    #    x0=data[i,j]
                else:
                    x0=data[i,j]
                exp_ret[i,j] = point_solver(locdata,window,j,logreturn=False,is_returns=True,x0=x0)
    dev_ret = data - exp_ret
    #sgn = np.sign(data*exp_ret)
    out = (dev_ret)*sdinv
    #out = sgn*abs(dev_ret)*sdinv
    out[0,:] = 0
    return out

###############
###         ###
### EXAMPLE ###
###         ###
###############

# =============================================================================
# import pandas as pd
# data = pd.read_csv(r'C:\temp\ad_test_input.csv', index_col=0) # TS in dataframe in wide format
# data.iloc[28:36,4]=data.iloc[28:36,4]*2/3
# data.iloc[12,0]=1186
# data.iloc[33,3]=data.iloc[33,3]*5/6
# data.iloc[34,3]=data.iloc[34,3]*5/6
# 
# import time
# start = time.time()
# out,outAF = anomalies_main(data.values)
# end = time.time()
# print(end - start)
# top_anomalies = pd.DataFrame(anomalies_sort2d(out,outAF))
# top_anomalies['MKF'] = (data.columns)[top_anomalies['MKF']]
# top_anomalies['Date'] = (data.index)[top_anomalies['Date']]
# 
# start = time.time()
# outES = exp_score(data.values)
# end = time.time()
# print(end - start)
# top_anomaliesES = pd.DataFrame(anomalies_sort(outES))
# top_anomaliesES['MKF'] = (data.columns)[top_anomaliesES['MKF']]
# top_anomaliesES['Date'] = (data.index)[top_anomaliesES['Date']]  
#
# #or use below for timing of main
# #%timeit anomalies_main(data.values)
# #%timeit exp_score(data.values)
# =============================================================================
