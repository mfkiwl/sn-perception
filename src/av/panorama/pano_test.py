import cv2
#from pathlib import Path
import os

import numpy as np
import time


# testing utility 

def resize_img(cv_img: np.ndarray, width: int, height: int) -> np.ndarray:
	"""Resize an OpenCV image with the given parameters."""
	cv_img = cv2.resize(
		cv_img, (width, height), interpolation=cv2.INTER_AREA
	)
	# cv_img = cv_img[..., np.newaxis]
	return cv_img


class CylindricalStitcher:
    """
    Cylindrical panorama stitcher.
    ARGS
    ----
        ctr ( Tuple[int, int] ):
            Original image center point coordinates (xc, yc). 
        f ( int ):
            Pinhole camera focal length (acquire value
            by calculating camera calibration matrix).
    """
    def __init__(self, height, width, f=1100):

        self.f = f

        self.h = height
        self.w = width

        self.ii_x = None
        self.ii_y = None
        self.it_tl_x = None
        self.it_tl_y = None
        self.weight_tl = None
        self.weight_tr = None
        self.weight_bl = None
        self.weight_br = None

        # min(ii_x)
        self.min_x = None
        # ii_x - min_x
        self.mask_x = None
        # ii_y
        self.mask_y = None

        # calculates values above
        self.calculate_cylinder_values()

        self.H_1 = np.array( [[1.80597322e+00, 1.01224729e-02, -5.22485146e+02],
                              [3.70355592e-01, 1.63199424e+00, -1.94327823e+02],
                              [1.09934460e-03, 2.42769073e-04, 1.00000000e+00]])

        self.H_2 = np.array([[ 4.64334213e-01, 1.46635317e-01, 8.06533224e+02],
                             [-2.36055664e-01, 9.86707286e-01, 2.22846913e+02],
                             [-4.37302593e-04, 1.21728684e-04, 1.00000000e+00]])
        
        self.new_height_l = 588
        self.new_width_l = 1130
        self.corr_l = [522, 194]

        self.H_l_corr = np.array( [[2.37983138e+00, 1.36847940e-01, -4.85146105e-01],
                                   [5.83628516e-01, 1.67909159e+00, -3.27823013e-01],
                                   [1.09934488e-03, 2.42769196e-04, 1.00000000e+00]])

        self.new_height_r = 588
        self.new_width_r = 1482
        self.corr_r = [0, 0]

        self.H_r_corr = np.array([[ 4.64334171e-01, 1.46635326e-01, 8.06533203e+02],
                                  [-2.36055661e-01, 9.86707203e-01, 2.22846909e+02],
                                  [-4.37302606e-04, 1.21728646e-04, 1.00000000e+00]])

    def calculate_cylinder_values(self):
        """
        Calculates cylinder projection values and
        saves them into class variables. Called 
        in the constructor.
        """

        ctr = (self.w // 2, self.h // 2)

        # storing all coordinates (x - y) of initial image (calculate once if size known?)
        coords_init = np.array([np.array([i, j]) for i in range(self.w) for j in range(self.h)])
        ii_x = coords_init[:, 0]
        ii_y = coords_init[:, 1]
    
        # Finding corresponding coordinates of the transformed image in the initial image
        it_x, it_y = self.unroll_x_y(ii_x, ii_y, ctr)

        # Rounding off the coordinate values to get exact pixel values (top-left corner) - tl
        it_tl_x = it_x.astype(int)
        it_tl_y = it_y.astype(int)

        # Finding transformed image points whose corresponding 
        # initial image points lies inside the initial image
        good_indices = (it_tl_x >= 0) * (it_tl_x <= (self.w-2)) * (it_tl_y >= 0) * (it_tl_y <= (self.h-2))

        # Removing all the outside points from everywhere
        self.ii_x = ii_x[good_indices]
        self.ii_y = ii_y[good_indices]

        it_x = it_x[good_indices]
        it_y = it_y[good_indices]

        self.it_tl_x = it_tl_x[good_indices]
        self.it_tl_y = it_tl_y[good_indices]

        # Bilinear interpolation
        dx = it_x - self.it_tl_x
        dy = it_y - self.it_tl_y

        self.weight_tl = (1.0 - dx) * (1.0 - dy)
        self.weight_tr = (dx)       * (1.0 - dy)
        self.weight_bl = (1.0 - dx) * (dy)
        self.weight_br = (dx)       * (dy)

        self.min_x = min(self.ii_x)
        self.mask_x = self.ii_x - self.min_x
        self.mask_y = self.ii_y

        print("[INFO] Values precalculated!")
    

    def project_onto_cylinder_preset(self, img_init):
        """
        Projects image onto a cylinder with
        certain dimensions (h, w) and focal
        length parameter (f).
        """
        # creating blank transformed image
        img_tr = np.zeros( (self.h, self.w, 3), dtype=np.uint8)

        img_tr[self.ii_y, self.ii_x, :] = ( self.weight_tl[:, None] * img_init[self.it_tl_y,     self.it_tl_x,     :] ) + \
                                    ( self.weight_tr[:, None] * img_init[self.it_tl_y,     self.it_tl_x + 1, :] ) + \
                                    ( self.weight_bl[:, None] * img_init[self.it_tl_y + 1, self.it_tl_x,     :] ) + \
                                    ( self.weight_br[:, None] * img_init[self.it_tl_y + 1, self.it_tl_x + 1, :] )
        
        min_x = min(self.ii_x)
        img_tr = img_tr[:, min_x : -min_x, :]

        return img_tr


    # TODO: Finish image reader for debugging.
    def read_images(self, img_dir_path):
        """
        This function is for debugging with images.
        """
        Images = []									# Input Images will be stored in this list.

        # Checking if path is of folder.
        if os.path.isdir(img_dir_path):                              # If path is of a folder contaning images.
            ImageNames = os.listdir(img_dir_path)
            ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in ImageNames]
            ImageNames_Split = sorted(ImageNames_Split, key=lambda x:x[0])
            ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]
            
            for i in range(len(ImageNames_Sorted)):                     # Getting all image's name present inside the folder.
                ImageName = ImageNames_Sorted[i]
                InputImage = cv2.imread(img_dir_path + "/" + ImageName)  # Reading images one by one.
                
                InputImage = resize_img(InputImage, self.w, self.h)

                # Checking if image is read
                if InputImage is None:
                    print("Not able to read image: {}".format(ImageName))
                    exit(0)

                Images.append(InputImage)                               # Storing images.
                
        else:                                       # If it is not folder(Invalid Path).
            print("\nEnter valid Image Folder Path.\n")
            
        if len(Images) < 2:
            print("\nNot enough images found. Please provide 2 or more images.\n")
            exit(1)
        
        return Images


    def unroll_x_y(self, x, y, center):
        """
        Calculate x' and y' (coordinates of an xy point
        in the unrolled cylindrical image). from
        x and y in the original image.
        RETURNS
        -------
        xt, yt (Tuple[int, int]):
            The transformed x' and y' points.
        """
        xt = self.f * np.tan( (x - center[0]) / self.f ) + center[0]
        yt = (y - center[1]) / np.cos( (x - center[0]) / self.f ) + center[1]

        return xt, yt


    def project_onto_cylinder(self, img_init):
        
        h, w = img_init.shape[:2]
        ctr = (w // 2, h // 2)

        # creating blank transformed image
        img_tr = np.zeros(img_init.shape, dtype=np.uint8)

        # storing all coordinates (x - y) of initial image (calculate once if size known?)
        coords_init = np.array([np.array([i, j]) for i in range(w) for j in range(h)])
        ii_x = coords_init[:, 0]
        ii_y = coords_init[:, 1]
    
        # Finding corresponding coordinates of the transformed image in the initial image
        it_x, it_y = self.unroll_x_y(ii_x, ii_y, ctr)

        # Rounding off the coordinate values to get exact pixel values (top-left corner) - tl
        it_tl_x = it_x.astype(int)
        it_tl_y = it_y.astype(int)

        # Finding transformed image points whose corresponding 
        # initial image points lies inside the initial image
        good_indices = (it_tl_x >= 0) * (it_tl_x <= (w-2)) * (it_tl_y >= 0) * (it_tl_y <= (h-2))

        # Removing all the outside points from everywhere
        ii_x = ii_x[good_indices]
        ii_y = ii_y[good_indices]

        it_x = it_x[good_indices]
        it_y = it_y[good_indices]

        it_tl_x = it_tl_x[good_indices]
        it_tl_y = it_tl_y[good_indices]

        # Bilinear interpolation
        dx = it_x - it_tl_x
        dy = it_y - it_tl_y

        weight_tl = (1.0 - dx) * (1.0 - dy)
        weight_tr = (dx)       * (1.0 - dy)
        weight_bl = (1.0 - dx) * (dy)
        weight_br = (dx)       * (dy)
        
        # calculate forward until this point if we have a determined size and focal length?

        img_tr[ii_y, ii_x, :] = ( weight_tl[:, None] * img_init[it_tl_y,     it_tl_x,     :] ) + \
                                         ( weight_tr[:, None] * img_init[it_tl_y,     it_tl_x + 1, :] ) + \
                                         ( weight_bl[:, None] * img_init[it_tl_y + 1, it_tl_x,     :] ) + \
                                         ( weight_br[:, None] * img_init[it_tl_y + 1, it_tl_x + 1, :] )

        # calculate these from size! (experimental)
        # img_tr, ii_x, ii_y, it_tl_x, it_tl_y, weight_tl, weight_tr, weight_bl, weight_br

        # Getting x coorinate to remove black region from right and left in the transformed image
        min_x = min(ii_x)

        # Cropping out the black region from both sides (using symmetricity)
        img_tr = img_tr[:, min_x : -min_x, :]

        return img_tr, ii_x-min_x, ii_y


    def pairwise_match(self):
        pass

    def stitcher(self, debug=False):
        pass


    def FindMatches(self, BaseImage, SecImage):
        # Using SIFT to find the keypoints and decriptors in the images
        Sift = cv2.SIFT_create()
        BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
        SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

        # Using Brute Force matcher to find matches.
        BF_Matcher = cv2.BFMatcher()
        InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

        # Applytng ratio stitcher and filtering out the good matches.
        GoodMatches = []
        for m, n in InitialMatches:
            if m.distance < 0.75 * n.distance:
                GoodMatches.append([m])

        return GoodMatches, BaseImage_kp, SecImage_kp


    def FindHomography(self, Matches, BaseImage_kp, SecImage_kp):
        # If less than 4 matches found, exit the code.
        if len(Matches) < 4:
            print("\nNot enough matches found between the images.\n")
            exit(0)

        # Storing coordinates of points corresponding to the matches found in both the images
        BaseImage_pts = []
        SecImage_pts = []
        for Match in Matches:
            BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
            SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

        # Changing the datatype to "float32" for finding homography
        BaseImage_pts = np.float32(BaseImage_pts)
        SecImage_pts = np.float32(SecImage_pts)

        # Finding the homography matrix(transformation matrix).
        (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

        return HomographyMatrix, Status


    def GetNewFrameSizeAndMatrix(self, HomographyMatrix, Sec_ImageShape, Base_ImageShape):
        # Reading the size of the image
        (Height, Width) = Sec_ImageShape
        
        # Taking the matrix of initial coordinates of the corners of the secondary image
        # Stored in the following format: [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]
        # Where (xt, yt) is the coordinate of the i th corner of the image. 
        InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                                [0, 0, Height - 1, Height - 1],
                                [1, 1, 1, 1]])
        
        # Finding the final coordinates of the corners of the image after transformation.
        # NOTE: Here, the coordinates of the corners of the frame may go out of the 
        # frame(negative values). We will correct this afterwards by updating the 
        # homography matrix accordingly.
        FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

        [x, y, c] = FinalMatrix
        x = np.divide(x, c)
        y = np.divide(y, c)

        # Finding the dimentions of the stitched image frame and the "Correction" factor
        min_x, max_x = int(round(min(x))), int(round(max(x)))
        min_y, max_y = int(round(min(y))), int(round(max(y)))

        New_Width = max_x
        New_Height = max_y
        Correction = [0, 0]
        if min_x < 0:
            New_Width -= min_x
            Correction[0] = abs(min_x)
        if min_y < 0:
            New_Height -= min_y
            Correction[1] = abs(min_y)
        
        # Again correcting New_Width and New_Height
        # Helpful when secondary image is overlaped on the left hand side of the Base image.
        if New_Width < Base_ImageShape[1] + Correction[0]:
            New_Width = Base_ImageShape[1] + Correction[0]
        if New_Height < Base_ImageShape[0] + Correction[1]:
            New_Height = Base_ImageShape[0] + Correction[1]

        # Finding the coordinates of the corners of the image if they all were within the frame.
        x = np.add(x, Correction[0])
        y = np.add(y, Correction[1])
        OldInitialPoints = np.float32([[0, 0],
                                    [Width - 1, 0],
                                    [Width - 1, Height - 1],
                                    [0, Height - 1]])
        NewFinalPonts = np.float32(np.array([x, y]).transpose())

        # Updating the homography matrix. Done so that now the secondary image completely 
        # lies inside the frame
        HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
        
        print(f"New Height: {New_Height}")
        print(f"New Width: {New_Width}")
        print(f"Correction: {Correction}\n")
        print(f"Corrected Homography: \n {HomographyMatrix} \n ")
        print("----------------------------------------\n")


        return [New_Height, New_Width], Correction, HomographyMatrix


    def StitchImages(self, BaseImage, SecImage):
        # Applying Cylindrical projection on SecImage
        SecImage_Cyl = self.project_onto_cylinder_preset(SecImage)

        # Getting SecImage Mask
        SecImage_Mask = np.zeros(SecImage_Cyl.shape, dtype=np.uint8)
        SecImage_Mask[self.mask_y, self.mask_x, :] = 255

        # Finding matches between the 2 images and their keypoints
        Matches, BaseImage_kp, SecImage_kp = self.FindMatches(BaseImage, SecImage_Cyl)
        
        # Finding homography matrix.
        HomographyMatrix, Status = self.FindHomography(Matches, BaseImage_kp, SecImage_kp)
        print(f"Homography, unchanged: \n {HomographyMatrix} \n")
        
        # Finding size of new frame of stitched images and updating the homography matrix 
        NewFrameSize, Correction, HomographyMatrix = self.GetNewFrameSizeAndMatrix(HomographyMatrix, SecImage_Cyl.shape[:2], BaseImage.shape[:2])

        # Finally placing the images upon one another.
        SecImage_Transformed = cv2.warpPerspective(SecImage_Cyl, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
        SecImage_Transformed_Mask = cv2.warpPerspective(SecImage_Mask, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
        BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
        BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

        StitchedImage = cv2.bitwise_or(SecImage_Transformed, cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))

        return StitchedImage


        # does focal length needs to be varied - if yes, create a variable focal length function
        # find out focal length of the AV camera LENS
        # try bigger AV images with ROS galactic


if __name__ == '__main__':
    # (384, 680) - outdoors: f=510 - indoors: f=540
    stitcher = CylindricalStitcher(384, 680, f=540)
    #img_init = cv2.imread("imgs/room/2.jpg")

    # Reading images.
    Images = stitcher.read_images("src/av/panorama/assets/indoors/room")
    print("[INFO] Images read!")

    start1 = time.process_time()
    tr_img = stitcher.project_onto_cylinder_preset(Images[0])

    print("New Version.")
    print(tr_img.shape)

    for i in range(1, len(Images)):
        StitchedImage = stitcher.StitchImages(tr_img, Images[i])

        tr_img = StitchedImage.copy() 
    stop1 = time.process_time() - start1

    # start2 = time.process_time()
    # tr_img, _, _ = stitcher.project_onto_cylinder_preset(img_init)
    # stop2 = time.process_time() - start2
    # start3 = time.process_time()
    # tr_img, _, _ = stitcher.project_onto_cylinder_preset(img_init)
    # stop3 = time.process_time() - start3

    print('Time 1: ', stop1)
    # print('Time 2: ', stop2)
    # print('Time 3: ', stop3)
    cv2.imwrite("Stitched_Panorama.png", tr_img)
    #cv2.imshow("image", tr_img)
    #cv2.waitKey(0)