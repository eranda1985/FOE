#Program to track optical flow  in video camera. 
import numpy as np
import time
import cv2
import foe_utils.foe_utils as fu
import background_model.background_model as bgmodel

cap = cv2.VideoCapture(0)

#parameters for Shi Tomasi corner detection
features = dict(maxCorners=100, qualityLevel = 0.3, minDistance=7, blockSize=7, mask=None)

#params for lucas kanade optical flow algorithm
lucas_kanade_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# initial frame grab
ret, first_frame = cap.read()

time1 = time.time()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = fu.calcfeatures(first_gray, features)
mask = np.zeros_like(first_frame)
intensity_samples = np.array([])
intensity_samples_y = np.array([])
ttc_frames_list = np.array([])

while(True):
    ret, second_frame = cap.read()
    diff = time.time() - time1
    time1 = time.time()
    framerate = int(1 /diff)
    #print('framerate: ' , framerate)
    second_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

    # if count of p0 is not sufficient get some more features.
    if(p0.shape[0] < 90):
        p0 = fu.calcfeatures(first_gray, features)
    if(np.any(p0) == False):
    	continue
    if(p0.shape[0] <= 1):
        p0 = fu.calcfeatures(first_gray, features)

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

    for i,(base,unit) in enumerate(zip(good_old, unit_vectors)):
        base = base.astype(int)
        x,y = base.ravel()
        mm = unit*100
        tan = mm[1]/mm[0]
        c = y - (tan*x)
        A = np.append(A, tan)
        C = np.append(C,c)

        mm = mm.astype(int)
        p,q = mm.ravel()
        mask = second_frame.copy()
        #mask = cv2.arrowedLine(second_frame, (x,y),(x+p,y+q), (0, 255, 0), 2, 8, 0, 0.5)           

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
            intersect = fu.seg_intersect(np.array([1.0, y1]), np.array([2.0, y2]), np.array([1.0, y3]), np.array([2.0, y4]))
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

        backgnd = bgmodel.background_model(second_frame)
        #backgndColor = cv2.cvtColor(backgnd, cv2.COLOR_GRAY2BGR)

        # Get the intensity of pixel at p,q
        if((0<= p < 480) and (0<=q < 640)):
            #intensity = second_gray[p][q]
            intensity_samples = np.append(intensity_samples, p)
            intensity_samples_y = np.append(intensity_samples_y, q)
            kalman = fu.kalman_filter(intensity_samples)
            kalman_y = fu.kalman_filter(intensity_samples_y)
            #print(kalman)
            foe_x = kalman[-1].astype(int)
            foe_y = kalman_y[-1].astype(int)
            if(foe_y >= 32 and foe_x >= 32):
                center_patch = backgnd[foe_y-32:foe_y+32, foe_x-32:foe_x+32]
                #cv2.imshow('center_patch', center_patch)
                #cv2.waitKey(1)
                whitePixels = np.sum(center_patch==255)
                blackPixels = np.sum(center_patch==0)
                if(blackPixels >= ((whitePixels+blackPixels)/5)):
                    mask = cv2.circle(second_frame, (foe_x,foe_y,), 10, (0, 0, 255), -1)
                    #backgndColor = cv2.circle(backgndColor, (foe_x,foe_y,), 10, (0, 0, 255), -1)
                else:
                    mask = cv2.circle(second_frame, (foe_x,foe_y,), 10, (0, 255, 255), -1)
                    #backgndColor = cv2.circle(backgndColor, (foe_x,foe_y,), 10, (0, 255, 255), -1)
            else:
                mask = cv2.circle(second_frame, (foe_x,foe_y,), 10, (0, 255, 255), -1)
                #backgndColor = cv2.circle(backgndColor, (foe_x,foe_y,), 10, (0, 255, 255), -1)
            foe = np.array([foe_x, foe_y]);

    cv2.imshow('frame',mask)
    #cv2.imshow('background', backgndColor)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
