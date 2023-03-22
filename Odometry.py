from glob import glob
import cv2, skimage, os
from scipy import signal, spatial
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.linalg
import argparse


parser = argparse.ArgumentParser(description='Visual Odometry.')
parser.add_argument('--version', choices=['manual', 'opencv'], 
                    help='One of "manual" or "opencv"')


class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]

        self.K = np.array([[self.focal_length, 0, self.pp[0]], [0, self.focal_length, self.pp[1]], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)

        self.imgs = np.array([self.imread(frame) for frame in self.frames]) # (n, h, w)
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def get_kp_data(self, img, detector_type='orb'):
        '''
        TODO:
        '''
        if detector_type == 'orb':
            detector = cv2.ORB_create()
            kp, des = detector.detectAndCompute(img, None)
            kp = np.array([k.pt for k in kp]) 
            return kp, des
        elif detector_type == 'fast':
            detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            kp = detector.detect(img)
            kp = np.array([x.pt for x in kp], dtype=np.float32).reshape(-1, 1, 2)
            return kp, None
        else:
            return None, None

    def get_best_matches(self, kp1, img2, num_matches, detector_type='orb'):
        '''
        TODO:
        '''
        kp1, des1 = kp1
        kp2, des2 = self.get_kp_data(img2, detector_type=detector_type)

        # Find distance between descriptors in images
        dist = spatial.distance.cdist(des1, des2, 'sqeuclidean')
        pairs = np.argsort(dist.reshape(-1))[:num_matches]
        out_kp = np.zeros((num_matches, 4))
        ind1, ind2 = np.unravel_index(pairs, dist.shape)
        out_kp[:,2:] = kp1[ind1]
        out_kp[:,:2] = kp2[ind2]
        
        return out_kp, (kp2, des2)

    def compute_inliers(self, est, threshold):
        '''
        TODO:
        '''
        # Find inliers
        # compute distance between origin and prediction
        dist = np.abs(est)
        # number of inliers are the sum of all points within threshold
        num_inliers = np.sum(dist < threshold)
        return dist, num_inliers

    def compute_residual(self, data, dist, threshold, num_inliers):
        '''Returns inliers and avg euclidean distance between inliers and origin.

        Args:
            TODO
        '''
        inliers = []
        avg_inlier_resid = 0.
        for i in range(len(dist)):
            if dist[i] < threshold:
                avg_inlier_resid += dist[i]
                inliers.append(list(data[i]))
        avg_inlier_resid /= num_inliers
        return avg_inlier_resid, inliers

    def ransac(self, data, max_iters=10000, min_inliers=10, initial_threshold=0.5):
        """
        Ransac code to find the best model (essential matrix), inliers, and residuals
        TODO
        """
        best_transform, best_inlier_count, avg_inlier_resid = None, 0, 0.
        inliers = []
        threshold = initial_threshold / self.focal_length
        
        # convert to homogenous coords (x,y) -> (x,y,1)
        homog_x = np.concatenate([data[:,:2], np.ones((len(data), 1))], axis=1)
        homog_xp = np.concatenate([data[:,2:], np.ones((len(data), 1))], axis=1)

        # Calibrate
        homog_x = (self.K_inv @ homog_x.T).T
        homog_xp = (self.K_inv @ homog_xp.T).T

        for i in range(max_iters):
            # Select random matches
            num_seeds = 8 # 8-point
            pts = np.random.randint(len(data), size=num_seeds)

            E = self.compute_E(data[pts], num_seeds=num_seeds)
            
            # Remove any non rank-2
            if np.linalg.matrix_rank(E) > 2: continue
            
            # Plug our E prediction to estimate Sampson's Error
            est = np.diag(homog_xp @ E @ homog_x.T)
            xp_E = homog_xp @ E
            x_E = (E @ homog_x.T).T
            est = est / np.sqrt( np.square(xp_E[:,0]) + np.square(xp_E[:,1])
                            + np.square(x_E[:,0]) + np.square(x_E[:,1]))
            est = np.square(est)

            # Compute Inliers
            dist, num_inliers = self.compute_inliers(est, threshold)

            # best transform is one with highest inlier_count
            if num_inliers > best_inlier_count:
                best_inlier_count = num_inliers
                best_transform = E
                avg_inlier_resid, inliers = self.compute_residual(data, dist, threshold, num_inliers)

            if num_inliers > min_inliers:
                break

        # Recompute with inliers
        E = self.compute_E(np.array(inliers), num_seeds=len(inliers))

        return best_transform, best_inlier_count, threshold, np.array(inliers)

    def compute_E(self, matches, num_seeds=8):
        """
        Compute fundamental matrix according to the matches
        TODO
        """
        x = matches[:,:2]
        x_prime = matches[:,2:]

        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        x_prime = np.concatenate([x_prime, np.ones((len(x_prime), 1))], axis=1)

        # Calibrate
        x = (self.K_inv @ x.T).T
        x_prime = (self.K_inv @ x_prime.T).T
        
        # Construct U
        U = np.zeros((num_seeds, 9))
        for i in range(num_seeds):
            x_i, y_i = x[i,0], x[i,1]
            xp_i, yp_i = x_prime[i,0], x_prime[i,1]
            U[i] = [x_i*xp_i, y_i*xp_i, xp_i,
                x_i*yp_i, y_i*yp_i, yp_i,
                x_i, y_i, 1]

        # Compute singular vector corresponding to smallest singular value of U^T U
        U, S, V = np.linalg.svd(U)
        E = V[-1].reshape(3,3)

        # Want E to be rank 2 so remove last singular vector then reassemble E
        U, S, V = np.linalg.svd(E)
        E = U @ np.diag([1,1,0]) @ V

        # Remove scale 
        E /= E[-1,-1]
        
        return E

    def triangulation(self, X1, X2, R, t, prev_Rt, X3=None, next_Rt=None, compute_cheirality=True):
        """
        TODO
        write your code to triangulate the points in 3D
        """
        # Compute projection Matrix from R, t, K
        Rt = np.concatenate([R, np.expand_dims(t, -1)], -1)
        P1 = Rt

        # Other projection is just [I | 0]
        if prev_Rt is None:
            P2 = np.eye(4)[:-1]
        else:
            P2 = prev_Rt
        
        X1 = np.concatenate([X1, np.ones((len(X1), 1))], axis=1)
        X2 = np.concatenate([X2, np.ones((len(X2), 1))], axis=1)

        # Calibrate
        X1 = (self.K_inv @ X1.T).T
        X2 = (self.K_inv @ X2.T).T

        # Use 3rd camera
        third_cam = False
        if next_Rt is not None:
            third_cam = True
            P3 = next_Rt
            X3 = np.concatenate([X3, np.ones((len(X3), 1))], axis=1)
            X3 = (self.K_inv @ X3.T).T

        X = np.zeros((X1.shape[0], 4))
        for i in range(X1.shape[0]):
            # Rewrite image coords as cross product, only need top 2 rows as bottom is not independent
            x_1 = np.array([
                [0, -1, X1[i,1]],
                [1, 0, -X1[i,0]],
            ])
            x_2 = np.array([
                [0, -1, X2[i,1]],
                [1, 0, -X2[i,0]],
            ])
            if third_cam:
                x_3 = np.array([
                    [0, -1, X3[i,1]],
                    [1, 0, -X3[i,0]],
                ])
                # Concatenate our projection from all images
                A = np.concatenate([x_2@P1, x_1@P2, x_3@P3], 0)

            else:
                # Concatenate our projection from both images
                A = np.concatenate([x_2@P1, x_1@P2], 0)

            # Solve for 3d point X
            U,S,V = np.linalg.svd(A)
            X[i] = V[-1]

        # Convert back from homogenous coords
        X = X / np.expand_dims(X[:,-1], -1)

        if compute_cheirality:
            # Check which points are in front of both cameras
            cheirality = ((P1 @ X.T).T[:,2] > 0) * (P2 @ X.T).T[:,2] > 0

            return np.sum(cheirality), cheirality
        else:
            return X

    def compute_Rt(self, E, prev_Rt, scale, inliers):
        '''
        TODO
        '''
        # https://en.wikipedia.org/wiki/Essential_matrix
        U, S, V = np.linalg.svd(E)
        S = np.array([1,1,0])
        Wp = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        t = V[-1]
        # (R, t, cheirality, cheirality inliers)
        best_Rt = (None, None, 0, [])

        if np.linalg.det(U) < 0:
            U *= -1
        if np.linalg.det(V) < 0:
            V *= -1

        for W in [Wp, Wp.T]:
            R = U @ W @ V
            # Ignore with |R| != 1
            if not np.isclose(np.linalg.det(R), 1): continue

            # Pair R with +/- t and select one with most pairs of points in front of both cameras
            for t_hat in [t, -t]:
                cheirality, cheirality_in = self.triangulation(inliers[:min(100,len(inliers)),:2], 
                                                               inliers[:min(100,len(inliers)),2:], 
                                                               R, t_hat, None)

                if cheirality > best_Rt[2]:
                    best_Rt = (R, t_hat, cheirality, cheirality_in)

        R, t, cheirality, cheirality_in = best_Rt

        if R is None:
            # Need to recompute E
            return None, None, None 

        # Make R/t global, currently w.r.t. past frame
        prev_t, prev_R = prev_Rt[:-1,-1:], prev_Rt[:-1,:-1]
        t = prev_t + scale*prev_R.dot(np.expand_dims(t, -1)) 
        R = prev_R.dot(R) 
        Rt = np.concatenate([R, t], axis=-1)
        Rt = np.concatenate([Rt, np.array([[0,0,0,1]])], axis=0)

        return Rt, cheirality, cheirality_in

    def smooth_pts(self, pts):
        '''TODO'''
        new_pts = np.zeros(pts.shape)
        window = 2
        for i in range(window+1):
            new_pts[i] = pts[i]
        for i in range(window+1, len(pts)-window):
            new_pts[i] = np.average(pts[i-window:i+window], axis=0)
        for i in range(len(pts)-window,len(pts)):
            new_pts[i] = pts[i]

        return new_pts

    def run_opencv(self):
        """
        TODO
        """
        # Starting Rt
        Rts = [np.eye(4)]
        pts = [[0,0,0]]

        for i in range(1,len(self.imgs)):

            if i == 1 or len(data_2) < 2000:
                print(f'Recomputing KPs {i}')
                kp_1, _ = self.get_kp_data(self.imgs[i-1], detector_type='fast')
            
            kp_2, st, err = cv2.calcOpticalFlowPyrLK(self.imgs[i-1], self.imgs[i], kp_1, None)
            # Save the good points from the optical flow
            data_1 = kp_1[st == 1]
            data_2 = kp_2[st == 1]

            E, _ = cv2.findEssentialMat(data_2, data_1, focal=self.focal_length, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=0.0003)
            _, R, t, mask = cv2.recoverPose(E, data_2, data_1, focal=self.focal_length, pp=self.pp)
            inliers = np.concatenate([data_2[np.squeeze(mask>0)], data_1[np.squeeze(mask>0)]], axis=-1)

            prev_t, prev_R = Rts[-1][:-1,-1:], Rts[-1][:-1,:-1]
            t = prev_t + self.get_scale(i)*prev_R.dot(t) 
            R = prev_R.dot(R) 
            Rt = np.concatenate([R, t], axis=-1)
            Rt = np.concatenate([Rt, np.array([[0,0,0,1]])], axis=0)

            Rts.append(Rt)
            pts.append(t.squeeze())
            kp_1 = kp_2

        return np.array(pts)
    
    def run(self):
        """
        TODO
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        # Starting Rt
        Rts = [np.eye(4)]
        pts = [[0,0,0]]

        # First image kps
        kps = [self.get_kp_data(self.imgs[0], detector_type='orb')]
        all_inliers = []

        for i in range(1,len(self.imgs)):
            
            # Get matches between new image and previous 
            data, img_kps = self.get_best_matches(kps[-1], self.imgs[i], 300, detector_type='orb')
            kps.append(img_kps)

            # Loop to ensure we get a reasonable E/R/t estimation
            Rt = None
            while Rt is None:
                # Ransac to estimate E
                E, max_inliers, threshold, inliers = self.ransac(data, max_iters=2000, min_inliers=60, initial_threshold=0.0001)
                # Use E to estimate R/t
                Rt, cheirality, cheirality_in = self.compute_Rt(E, Rts[-1], self.get_scale(i), inliers)            
            
            all_inliers.append(inliers)
            Rts.append(Rt)
            pts.append(Rt[:-1,-1])

        # Smooth trajectory
        pts = self.smooth_pts(np.array(pts))
        return pts


def display(pred_pts):
    '''
    TODO
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    with open('./video_train/gt_sequence.txt', 'r') as f:
        gt_pts = [line.strip().split() for line in f.readlines()]

    traj = np.zeros(shape=(600, 1200, 3))
    mse_scores = []

    for gt, pred in zip(gt_pts, pred_pts):
        gt = np.array([float(gt[3]), float(gt[7]), float(gt[11])])
        
        mse_scores.append(np.linalg.norm(pred - gt))

        draw_x, draw_y, draw_z = [int(round(x)) for x in pred]
        true_x, true_y, true_z = [int(round(x)) for x in gt]

        traj = cv2.circle(traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
        traj = cv2.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

    cv2.putText(traj, 'Actual Position:', (140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv2.putText(traj, '-', (270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
    cv2.putText(traj, 'Estimated Odometry Position:', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv2.putText(traj, '-', (270, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

    cv2.imshow('trajectory', traj)
    cv2.imwrite(f'./trajectory-{args.version}.png', traj)

    plt.plot(mse_scores)
    plt.savefig(f'./mse-{args.version}.png')


if __name__=="__main__":
    args = parser.parse_args()

    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)

    start = time.time()
    if args.version == 'opencv':
        path = odemotryc.run_opencv()
    elif args.version == 'manual':
        path = odemotryc.run()
    print(f'Time: {int(time.time()-start)} s')

    np.save(f'predictions-{args.version}.npy', path)
    display(path)