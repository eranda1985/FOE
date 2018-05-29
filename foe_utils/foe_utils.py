import numpy as np
import cv2 

# Returns shi-tomasi features
def calcfeatures(first_gray, configs):
    featureElems = cv2.goodFeaturesToTrack(first_gray, **configs)
    return featureElems

# Method for checking vector similarity 
def cos_similarity(v0, v1):
    dot_product = np.dot(v0, v1)
    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)
    return dot_product/(norm_v0 * norm_v1)

# Returns the perpendicular vector for a given vector
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

# Method for Kalman Filter
def kalman_filter(samples):
    # intial parameters
    n_iter = samples.size
    sz = (n_iter,)

    Q = 1e-5 # process variance

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    R = 0.1**2 # estimate of measurement variance, change to see effect
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(samples[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    return xhat 