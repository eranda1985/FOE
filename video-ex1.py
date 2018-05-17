#Program to track optical flow  in video camera. 
import numpy as np
import cv2

def CalcFeatures(first_gray, configs):
    featureElems = cv2.goodFeaturesToTrack(first_gray, **configs)
    return featureElems

def Cos_Similarity(v0, v1):
    dot_product = np.dot(v0, v1)
    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)
    return dot_product/(norm_v0 * norm_v1)

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def Kalman_Filter(samples):
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

cap = cv2.VideoCapture(0)

#parameters for Shi Tomasi corner detection
features = dict(maxCorners=100, qualityLevel = 0.3, minDistance=7, blockSize=7, mask=None)

#params for lucas kanade optical flow algorithm
lucas_kanade_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
ret, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = CalcFeatures(first_gray, features)
mask = np.zeros_like(first_frame)
unit_vectors_old = np.array([])
sumx_old = np.array([])
intensity_samples = np.array([])
intensity_samples_y = np.array([])

while(True):
    print('------------------------------------------------------------------------------')
    ret, second_frame = cap.read()
    second_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

    # if count of p0 is not sufficient get some more features.
    if(p0.shape[0] < 90):
        p0 = CalcFeatures(first_gray, features)
    if(p0.shape[0] <= 1):
        p0 = CalcFeatures(first_gray, features)

    p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray, second_gray, p0, None, **lucas_kanade_params)
    
    good_new = p1[st==1]
    good_old = p0[st==1]

    displacement = good_new - good_old
    magnitude = np.sqrt((displacement**2).sum())[..., np.newaxis]

    #normalize
    unit_vectors = (displacement/magnitude)

    first_gray = second_gray.copy()
    p0 = good_new.reshape(-1, 1,2)

    A = np.array([])
    C = np.array([])

    for i,(base,unit,unit_old) in enumerate(zip(good_old, unit_vectors,unit_vectors_old)):
        base = base.astype(int)
        x,y = base.ravel()
        #mm = np.array([0,0])
        
        #if(unit_old.size > 0 and Cos_Similarity(unit_old, unit) > 0.9):
        mm = unit*100
        tan = mm[1]/mm[0]
        c = y - (tan*x)
        A = np.append(A, tan)
        C = np.append(C,c)

        mm = mm.astype(int)
        p,q = mm.ravel()
        mask = cv2.arrowedLine(second_frame, (x,y),(x+p,y+q), (0, 255, 0), 2, 8, 0, 0.5)           

    A = np.vstack([A, -1*np.ones(len(A))]).T
    C = C.reshape(-1,1)

    points = np.array([])
    points_y = np.array([])

    if(A.size > 0):
        start_a = A[0][0]
        start_c = C[0][0]

        for i,(a,c) in enumerate(zip(A,C)):
            if(start_a == a[0]):
                continue

            #x1 = 1.0
            y1 = start_a*(1.) + start_c
            #x2 = 2.0
            y2 = start_a*(2.) + start_c
            #x3 = 1.0
            y3 = a[0]*(1.) + c[0]
            #x4 = 2.0
            y4 = a[0]*(2.) + c[0]
            intersect = seg_intersect(np.array([1.0, y1]), np.array([2.0, y2]), np.array([1.0, y3]), np.array([2.0, y4]))
            points = np.append(points, intersect[0])
            points_y = np.append(points_y, intersect[1])

            #print(intersect)
            x,y = intersect.astype(int).ravel()
            #mask = cv2.circle(mask, (x,y,), 5, (0, 0, 255), -1)

        hist, edges = np.histogram(points,10)
        hist_y, edges_y = np.histogram(points_y,10)

        idx = (np.argmax(hist))
        idx_y = (np.argmax(hist_y))

        f_points = points[np.logical_and(points>=edges[idx], points<=edges[idx+1])].astype(int)
        f_points_y = points_y[np.logical_and(points_y>=edges_y[idx_y], points_y<=edges_y[idx_y+1])].astype(int)

        p = np.mean(f_points).astype(int)
        q = np.mean(f_points_y).astype(int)
        
        # Get the intensity of pixel at p,q
        if((0<= p < 480) and (0<=q < 640)):
            #intensity = second_gray[p][q]
            intensity_samples = np.append(intensity_samples, p)
            intensity_samples_y = np.append(intensity_samples_y, q)
            kalman = Kalman_Filter(intensity_samples)
            kalman_y = Kalman_Filter(intensity_samples_y)
            #print(kalman)
            foe_x = kalman[-1].astype(int)
            foe_y = kalman_y[-1].astype(int)
            mask = cv2.circle(second_frame, (foe_x,foe_y,), 10, (0, 0, 255), -1)
            foe = np.array([foe_x, foe_y]);
            d_from_foe = good_new - foe;
            print(d_from_foe/unit_vectors)
            #d_from_foe_magnitude = np.sqrt((d_from_foe**2).sum())[..., np.newaxis]

        #mask = cv2.circle(second_frame, (p,q,), 10, (0, 0, 255), -1)

    cv2.imshow('frame',mask)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

    unit_vectors_old = unit_vectors.copy()

cap.release()
cv2.destroyAllWindows()