from glob import glob
import cv2, skimage, os
import numpy as np
from scipy.optimize import least_squares

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            self.P = np.array([[self.focal_length,0,self.pp[0],0],[0,self.focal_length,self.pp[1],0],[0,0,1,0]],dtype=np.float64)
            self.K = self.P[:,:3]
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
     
    
    
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
        
    
    def get_gt_mat(self, frame_id):
        
        pose = np.array(self.pose[frame_id],dtype = np.float64)#added
        pose = pose.reshape(3, 4)#added
        pose = np.vstack((pose, [0, 0, 0, 1]))#added
        
        #x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        #return np.array([[x], [y], [z]])
        return pose
    
    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def combine_Rt(self, R, t):
        """
        Creates a transformation matrix by combining Rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray) : Rotation matrix  (3x3)
        t (ndarray) : translation vector (3x1)
    
        Returns
        -------
        T (ndarray) : Transformation matrix (4x4)

        """
        T = np.eye(4, dtype = np.float64)
        T[:3, :3] = R #upper left 3x3 matrix is rotation matrix
        T[:3, 3] = t #3x1 vector on right side is translation vector
        
        return T
        
    def get_matches(self, frame_id):
        """
        Function computes the matching keypoints between i-1 and i frame
        
        Parameters
        ----------
        cframe (int): current frame
        
        Returns
        -------
        q1 (ndarray): keypoint match position in i-1 image
        q2 (ndarray): keypoint match position in i image
        """
        # Get keypoints and feature descriptors using ORB detector algorithm
        orb = cv2.ORB_create(nfeatures = 4000)
        kp1, des1 = orb.detectAndCompute(self.imread(self.frames[frame_id - 1]), None)
        kp2, des2 = orb.detectAndCompute(self.imread(self.frames[frame_id]), None)
        
        # Brute Force Matching
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        # matches = bf.match(des1, des2)
        
        #Flann based matcher used to match descriptors between frames
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6,
                            key_size = 10,
                            multi_probe_level = 1)
        search_params = dict(checks = 150)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Only need the good matches, so we can filter out larger numbers
        #Lowe's ratio test
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        except ValueError: #skip over NAN values
            pass
        
        #Get q1, q2 from good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])  
        
        
        return q1, q2

    def get_transformation(self, q1, q2, frame_id):
        """
        Creates Transformation matrix

        Parameters
        ----------
        q1 (ndarray) : Keypoint match positions in i-1 image
        q2 (ndarray) : Keypoint match positions in i image

        Returns
        -------
        T_mat (ndarray) : Transformation matrix

        """
        #Find essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(q1, q2, self.K, method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
        
        #Decompose Essential matrix into R (rotation) and t (translation)
        R, t = self.decomp_E_mat(E, q1, q2)
        t = t * self.get_scale(frame_id)
        
        #Combine R and t to form transormation matrix
        T_mat = self.combine_Rt(R,t)
        
                
        return T_mat
    
    def reproj_loss(self, proj_3dpoints, pts_2d, num_pts):
        """
        Reprojection loss. Defines the error in projection matrix

        Parameters
        ----------
        proj_3dpoints (ndarray) : Camera projection and all 3D pooints
        2d_pts (ndarray): 2d locations of matches
        num_pts (int): number of points in sequence

        Returns
        -------
        error (ndarray)

        """
        P = proj_3dpoints[0:12].reshape(3,4)
        pts_3d = proj_3dpoints[12:].reshape((num_pts,4))
        
        reproj_error = []
        
        for idx, pt_3d in enumerate(pts_3d):
            pt_2d = np.array([pts_2d[0][idx], pts_2d[1][idx]])
            
            reproj_pt = np.matmul(P, pt_3d)
            reproj_pt /= reproj_pt[2]
        
            reproj_error.append(pt_2d - reproj_pt[0:2])
            
        return np.array(reproj_error).ravel()
    
    def bundle_adjustment(self, pts_3d, pts_2d, img, proj_mat):
        """
        Performs bundle adjustment by minimizing reprojection error with least squares
        Parameters
        ----------
        pts_3d (ndarray) : old 3d coordinate predictions
        pts_2d (ndarray): matching coordinates
        img (ndarray): current img in frame
        proj_mat (ndarray): computed projection matrix (3x4)

        Returns
        -------
        P (ndarray): New projection matrix after adjustment (3x4)
        pts_3d (ndarray): new coordinates after adjustment

        """
        proj_3dpoints = np.hstack((proj_mat.ravel(), pts_3d.ravel(order = "F")))
        num_pts = len(pts_2d[0])
        
        corrected = least_squares(self.reproj_loss, proj_3dpoints, args = (pts_2d, num_pts))
        
        P = corrected.x[0:12].reshape(3,4)
        pts_3d = corrected.x[12:].reshape((num_pts, 4))
        
        return P, pts_3d
    
    
    
    
    def decomp_E_mat(self, E, q1, q2):
        """
        Decomposes E matrix in order to find the correct R and t transformation
        
        Parameters
        ----------
        E (ndarry) : Essential matrix computed using RANSAC
        q1 (ndarray) : Keypoint match locations from i-1 image
        q2 (ndarray) : keypoint match location from i image

        Returns
        -------
        R (ndarry) : Rotation matrix
        t (ndarray) : translation vector

        """
        def z_calib(R,t):
            
            T = self.combine_Rt(R,t)
            
            P = np.matmul(np.concatenate((self.K, np.zeros((3,1))), axis=1),T)
            
            #Homogeneous Q for both frames
            hQ1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hQ2 = np.matmul(T, hQ1)
            
            #Get out of homogeneous coorde
            Q1 = hQ1[:3, :] / hQ1[3, :]
            Q2 = hQ2[:3, :] / hQ2[3, :]
            
            #sum of positive entries of Q
            sum_pos_z1 = sum(Q1[2,:] > 0)
            sum_pos_z2 = sum(Q2[2,:] > 0)
            pos_z = sum_pos_z1 + sum_pos_z2
            
            
            return pos_z           
        
              
        #built in OpenCV function
        #OpenCV function creates four possible R,t combinations, and t is not scaled
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t) #ensure correct array dim
        
        #List potential combinations of R and t
        combinations = [[R1,t],[R1,-t],[R2,t],[R2,-t]]
        
        #Get the correct combination
        sum_zs = []

        for R,t in combinations:
            sum_z = z_calib(R,t)
            sum_zs.append(sum_z)
 
            
        #Find the best combination - one with most positive z
        best_comb_idx = np.argmax(sum_zs)
        best_comb = combinations[best_comb_idx]

        R,t = best_comb

        
        
        return [R, t]

    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """

        path = np.zeros((len(self.frames),3))
        gt = np.zeros((len(self.frames),3))
        
        for frame in range(len(path)):
        #for frame in range(50):
            #gt_pose = np.concatenate((np.squeeze(self.get_gt(frame).T),[1]))
            gt_pose = self.get_gt_mat(frame)
            #print('GT',[gt_pose[0,3],gt_pose[1,3],gt_pose[2,3]])
            if frame == 0:
                cur_pose = gt_pose
            else:
                q1,q2 = self.get_matches(frame)
                transformation = self.get_transformation(q1,q2,frame)
                #print(transformation)
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transformation))
                #print('inverse of t',np.linalg.inv(transformation))
                #print('pose',cur_pose)
            path[frame] = [cur_pose[0,3],cur_pose[1,3],cur_pose[2,3]]
            gt[frame] = np.squeeze(self.get_gt(frame))

            #print('cur',cur_pose)
            print('GT',gt[frame])
            print('Est',path[frame])
        
        return path
        

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)