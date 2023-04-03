from glob import glob
import cv2, skimage, os
from scipy import signal, spatial
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse


parser = argparse.ArgumentParser(description='Visual Odometry.')
parser.add_argument('--version', choices=['manual', 'opencv'], 
                    help='One of "manual" or "opencv"')
parser.add_argument('--seq_path', type=str, help='Path to Kitti sequence.')
parser.add_argument('--pose_txt_path', type=str, help='Path to Kitti pose txt file.')


class ClassicalVO:

    def __init__(self, seq_path, pose_txt_path):
        self.seq_path = seq_path
        self.frames = sorted(glob(os.path.join(self.seq_path, 'image_0', '*')))
        # Read in calibration matrix
        with open(os.path.join(seq_path, 'calib.txt'), 'r') as f:
            line = f.readline().strip().split()[1:]
            self.focal_length = float(line[0])
            self.pp = (float(line[2]), float(line[6]))
        
        # Read in ground truth trajectory
        with open(os.path.join(pose_txt_path), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]

        # Set up calibration matrix
        self.K = np.array([[self.focal_length, 0, self.pp[0]], 
                           [0, self.focal_length, self.pp[1]], 
                           [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)

        # (N, H, W)
        self.imgs = np.array([cv2.imread(frame, 0) \
                              for frame in self.frames])

    def get_gt(self, frame_id):
        ''' Returns translation for GT pose corresponding to frame_id. '''
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        ''' Returns GT translation scale. '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def get_kp_data(self, img, detector_type='orb'):
        ''' Extract Keypoints from img using OpenCV detector_type. '''
        if detector_type == 'orb':
            detector = cv2.ORB_create()
            kp, des = detector.detectAndCompute(img, None)
            kp = np.array([k.pt for k in kp]) 
            return kp, des
        elif detector_type == 'fast':
            detector = cv2.FastFeatureDetector_create(threshold=25, 
                                                      nonmaxSuppression=True)
            kp = detector.detect(img)
            kp = np.array([x.pt for x in kp], dtype=np.float32)
            kp = kp.reshape(-1, 1, 2)
            return kp, None
        else:
            return None, None

    def get_best_matches(self, kp1, img2, n_matches, detector_type='orb'):
        '''
        Extracts keypoints on img2 then finds the closest n_matches matches
        with keypoints from img1 by feature descriptors.
        '''
        kp1, des1 = kp1
        kp2, des2 = self.get_kp_data(img2, detector_type=detector_type)

        # Find distance between descriptors in images
        dist = spatial.distance.cdist(des1, des2, 'sqeuclidean')
        pairs = np.argsort(dist.reshape(-1))[:n_matches]
        out_kp = np.zeros((n_matches, 4))
        ind1, ind2 = np.unravel_index(pairs, dist.shape)
        out_kp[:,2:] = kp1[ind1]
        out_kp[:,:2] = kp2[ind2]
        
        return out_kp, (kp2, des2)

    def compute_inliers(self, pred, threshold):
        '''
        Compute prediction distance and number of pred points 
        within threshold of origin.
        '''
        # Find inliers
        # compute distance between origin and prediction
        dist = np.abs(pred)
        # number of inliers are the sum of all points within threshold
        num_inliers = np.sum(dist < threshold)
        return dist, num_inliers

    def compute_residual(self, data, dist, threshold, num_inliers):
        '''
        Returns inliers and avg euclidean distance between 
        inliers and origin.
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
        Ransac to find the best model (essential matrix), inliers, and residuals
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
            n_seeds = 8 # 8-point
            pts = np.random.randint(len(data), size=n_seeds)

            E = self.compute_E(data[pts], n_seeds=n_seeds)
            
            # Remove any non rank-2
            if np.linalg.matrix_rank(E) > 2: continue
            
            # Plug our E prediction to estimate Sampson's Error
            pred = np.diag(homog_xp @ E @ homog_x.T)
            xp_E = homog_xp @ E
            x_E = (E @ homog_x.T).T
            pred = pred / np.sqrt( np.square(xp_E[:,0]) + np.square(xp_E[:,1])
                            + np.square(x_E[:,0]) + np.square(x_E[:,1]))
            pred = np.square(pred)

            # Compute Inliers
            dist, num_inliers = self.compute_inliers(pred, threshold)

            # best transform is one with highest inlier_count
            if num_inliers > best_inlier_count:
                best_inlier_count = num_inliers
                best_transform = E
                avg_inlier_resid, inliers = self.compute_residual(data, dist, threshold, num_inliers)

            if num_inliers > min_inliers:
                break

        # Recompute with inliers
        E = self.compute_E(np.array(inliers), n_seeds=len(inliers))

        return best_transform, best_inlier_count, threshold, np.array(inliers)

    def compute_E(self, matches, n_seeds=8):
        """
        Compute Essential matrix with normalized 8-point using matches.
        """
        x = matches[:,:2]
        x_prime = matches[:,2:]

        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        x_prime = np.concatenate([x_prime, np.ones((len(x_prime), 1))], axis=1)

        # Calibrate
        x = (self.K_inv @ x.T).T
        x_prime = (self.K_inv @ x_prime.T).T
        
        # Construct U
        U = np.zeros((n_seeds, 9))
        for i in range(n_seeds):
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

    def triangulate(self, X1, X2, R, t, prev_Rt, 
                      X3=None, 
                      next_Rt=None, 
                      compute_cheirality=True):
        """
        Triangulate 2D points from views 1/2 (X1/X2) in 3D.
        If X3 and next_Rt are not None then triangulates the
        points from all 3 views.

        Args:
            X1: 2D points from view 1
            X2: 2D points from view 2
            R: Rotation matrix (3,3)
            t: translation vector (3)
            prev_Rt: previous pose (3,3)
            X3: 2D points from view 3
            next_Rt: next pose (3,3)
            compute_cheirality: Whether to compute cheirality
        """
        # Compute projection Matrix from R, t
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
            # Rewrite image coords as cross product, 
            #   only need top 2 rows as bottom is not independent
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
        X = X / np.expand_dims(X[:,-1] + 1e-10, -1)

        if compute_cheirality:
            # Check which points are in front of both cameras
            cheirality = ((P1 @ X.T).T[:,2] > 0) * (P2 @ X.T).T[:,2] > 0

            return np.sum(cheirality), cheirality
        else:
            return X

    def compute_Rt(self, E, prev_Rt, scale, inliers):
        '''
        Estimate global pose from essential matrix E and previous pose prev_Rt.
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
                cheirality, cheirality_in = self.triangulate(inliers[:min(100,len(inliers)),:2], 
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
        '''Smooth pts with average of points within window (idx +/- 2). '''
        new_pts = np.zeros(pts.shape)
        window = 2
        for i in range(window+1):
            new_pts[i] = pts[i]
        for i in range(window+1, len(pts)-window):
            new_pts[i] = np.average(pts[i-window:i+window], axis=0)
        for i in range(len(pts)-window,len(pts)):
            new_pts[i] = pts[i]

        return new_pts
    

    def compute_sequence_ATE(self, poses):
        '''
        Returns the average ATE across all 3/5-frame snippets (to match SfMLearner)
        '''
        def convert_to_pose0_origin(pose):
            return np.linalg.inv(pose[0]) @ pose
        
        def compute_snippet_ATE(gt, pred):
            scale = np.sum(gt[:,:-1,-1] * pred[:,:-1,-1]) / np.sum(pred[:,:-1,-1] ** 2)
            snippet_ATE = np.linalg.norm((gt[:,:-1,-1] - scale * pred[:,:-1,-1]).reshape(-1)) 
            return snippet_ATE / pred.shape[0]
        
        # 3 frame snippets
        ATE_3frame = []
        for i in range(len(poses)-2):
            seq = np.stack([
                poses[i], poses[i+1], poses[i+2]
                ], axis=0)
            seq = convert_to_pose0_origin(seq)

            seq_gt = np.stack([
                np.concatenate([np.array(self.pose[i+j]).reshape(3,4).astype(np.float32),
                                np.array([[0,0,0,1]]),
                                ], axis=0)
                for j in range(3)], axis=0)
            seq_gt = convert_to_pose0_origin(seq_gt)

            snippet_ATE = compute_snippet_ATE(seq_gt, seq)
            ATE_3frame.append(snippet_ATE)

        # 5 frame snippets
        ATE_5frame = []
        for i in range(len(poses)-4):
            seq = np.stack([
                poses[i], poses[i+1], poses[i+2], poses[i+3], poses[i+4]
                ], axis=0)
            seq = convert_to_pose0_origin(seq)

            seq_gt = np.stack([
                np.concatenate([np.array(self.pose[i+j]).reshape(3,4).astype(np.float32),
                                np.array([[0,0,0,1]]),
                                ], axis=0)
                for j in range(5)], axis=0)
            seq_gt = convert_to_pose0_origin(seq_gt)

            snippet_ATE = compute_snippet_ATE(seq_gt, seq)
            ATE_5frame.append(snippet_ATE)
            
        return np.average(ATE_3frame), np.average(ATE_5frame)


    def run_opencv(self):
        '''
        Estimates ego-motion using OpenCV. Tracks features with FAST and Lucas Kanade 
        and re-estimates new features if < 2000 keypoints remain in frame. Estimates
        the Essential matrix with Nister's 5-Point and Ransac then extracts the relative 
        pose and converts to global.
        '''
        # Starting Rt
        Rts = [np.eye(4)]
        pts = [[0,0,0]]

        for i in range(1,len(self.imgs)):

            # Recompute keypoints
            if i == 1 or len(data_2) < 2000:
                kp_1, _ = self.get_kp_data(self.imgs[i-1], 
                                           detector_type='fast')
            
            kp_2, st, err = cv2.calcOpticalFlowPyrLK(self.imgs[i-1], 
                                                     self.imgs[i], 
                                                     kp_1, 
                                                     None)
            # Save the good points from the optical flow
            data_1 = kp_1[st == 1]
            data_2 = kp_2[st == 1]

            E, _ = cv2.findEssentialMat(data_2, data_1, 
                                        focal=self.focal_length, 
                                        pp=self.pp, 
                                        method=cv2.RANSAC, 
                                        prob=0.999, 
                                        threshold=0.0003)
            _, R, t, mask = cv2.recoverPose(
                E, data_2, data_1, focal=self.focal_length, pp=self.pp
                )
            inliers = np.concatenate([
                data_2[np.squeeze(mask>0)], data_1[np.squeeze(mask>0)]
                ], axis=-1)

            # Convert to global pose
            prev_t, prev_R = Rts[-1][:-1,-1:], Rts[-1][:-1,:-1]
            t = prev_t + self.get_scale(i)*prev_R.dot(t) 
            R = prev_R.dot(R) 
            Rt = np.concatenate([R, t], axis=-1)
            Rt = np.concatenate([Rt, np.array([[0,0,0,1]])], axis=0)

            Rts.append(Rt)
            pts.append(t.squeeze())
            kp_1 = kp_2

        return np.array(pts), np.array(Rts)
    
    def run_manual(self):
        """
        Estimates ego-motion manually (using OpenCV only for feature extraction). 
        Matches the 300 closest keypoints using orb features across image pairs. 
        Estimates the Essential matrix with Normalized 8-point algorithm and Ransac 
        then extracts the relative pose with the highest cheirality and converts to global.
        """
        # Starting Rt
        Rts = [np.eye(4)]
        pts = [[0,0,0]]

        # First image kps
        kps = [self.get_kp_data(self.imgs[0], detector_type='orb')]
        all_inliers = []

        for i in range(1,len(self.imgs)):
            
            # Get matches between new image and previous 
            data, img_kps = self.get_best_matches(kps[-1], 
                                                  self.imgs[i], 
                                                  300, 
                                                  detector_type='orb')
            kps.append(img_kps)

            # Loop to ensure we get a reasonable E/R/t estimation
            Rt = None
            while Rt is None:
                # Ransac to estimate E
                E, max_inliers, threshold, inliers = self.ransac(data, 
                                                                 max_iters=2000, 
                                                                 min_inliers=60, 
                                                                 initial_threshold=0.0001)
                # Use E to estimate R/t
                Rt, cheirality, cheirality_in = self.compute_Rt(E, 
                                                                Rts[-1], 
                                                                self.get_scale(i), 
                                                                inliers)            
            
            all_inliers.append(inliers)
            Rts.append(Rt)
            pts.append(Rt[:-1,-1])

        # Smooth trajectory
        pts = self.smooth_pts(np.array(pts))
        return pts, np.array(Rts)


def display(pred_pts):
    '''
    Display the ground truth compared to the estimated trajectory.
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    with open(args.pose_txt_path, 'r') as f:
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

    cv2.putText(traj, 'Actual Position:', (140, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv2.putText(traj, '-', (270, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
    cv2.putText(traj, 'Estimated Odometry Position:', (30, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv2.putText(traj, '-', (270, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

    cv2.imshow('trajectory', traj)
    cv2.imwrite(f'./trajectory-{args.version}.png', traj)

    plt.plot(mse_scores)
    plt.savefig(f'./mse-{args.version}.png')


if __name__=="__main__":
    args = parser.parse_args()

    odometry = ClassicalVO(args.seq_path, args.pose_txt_path)

    start = time.time()
    if args.version == 'opencv':
        path, poses = odometry.run_opencv()
    elif args.version == 'manual':
        path, poses = odometry.run_manual()

    print(f'Time: {int(time.time()-start)}s')

    ATE_3frame, ATE_5frame = odometry.compute_sequence_ATE(poses)
    print(f'ATE: {ATE_3frame} (3 Frame), {ATE_5frame} (5 Frame)')
    print(f'Path length: {len(poses)}')

    # Save global predictions and display video trajectory against GT
    np.save(f'predictions-{args.version}.npy', path)
    display(path)