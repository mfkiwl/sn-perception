import numpy as np
import cv2

import time
import os

# temporary
import imutils

class CylindricalStitcher:
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

        # precalculated homography matrices
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

        img_tr[self.ii_y, self.ii_x, :] = ( self.weight_tl[:, None] * img_init[self.it_tl_y, self.it_tl_x,   :] ) + \
                                    ( self.weight_tr[:, None] * img_init[self.it_tl_y,     self.it_tl_x + 1, :] ) + \
                                    ( self.weight_bl[:, None] * img_init[self.it_tl_y + 1, self.it_tl_x,     :] ) + \
                                    ( self.weight_br[:, None] * img_init[self.it_tl_y + 1, self.it_tl_x + 1, :] )
        
        img_tr = img_tr[:, self.min_x : -self.min_x, :]

        return img_tr


    def find_matches(self, base_img, sec_img):
        """
        Find image keypoints and descriptors,
        and find matches between them.
        """
        return good_matches, base_image_kp, sec_image_kp


    def find_homography(self, matches, base_img_kp, sec_img_kp):
        """
        Calculates homography between two images.
        """
        return homography_matrix, status


    def get_new_frame_size_and_matrix(self, homography_matrix, sec_img_shape, base_img_shape):
        """
        Corrects frame size and homography matrix.
        """
        return [new_height, new_width], correction, homography_matrix


    def stitch_images(self, base_img, sec_img, side, precalc=True):
        """Stitches two images together."""
        
        # project secondary image onto a cylinder
        sec_img_cyl = self.project_onto_cylinder_preset(sec_img)

        # get secondary image mask
        sec_img_mask = np.zeros(sec_img_cyl.shape, dtype=np.uint8)
        sec_img_mask[self.mask_y, self.mask_x, :] = 255

        if not precalc:
            # Find image keypoints and matches between them.
            matches, base_img_kp, sec_img_kp = self.find_matches(base_img, sec_img_cyl)

            # TODO: pass side to the function, so the homography
            # calculation and correction can be written into the 
            # right class variables (or make it more modular! along with warp)

            if side == 'left':
                pass
            elif side == 'right':
                pass
            else:
                print("Invalid side value. Valid values: 'left', 'right'")

            # Find homography matrix.
            homography_matrix, status = self.find_homography(matches, base_img_kp, sec_img_kp)

            # Finding size of new frame of stitched images and updating the homography matrix. 
            new_frame_size, correction, homography_matrix = self.get_new_frame_size_and_matrix(
                homography_matrix, sec_img_cyl.shape[:2], base_img.shape[:2]
            )

        # TODO: Precalculate and print new_frame size (for both, correction, and corrected homography matrix)

        if side == 'left':
            # Placing the images upon one another
            sec_img_transformed = cv2.warpPerspective(
                sec_img_cyl, self.H_l_corr, (self.new_width_l, self.new_height_l)
            )
            sec_img_transformed_mask = cv2.warpPerspective(
                sec_img_mask, self.H_l_corr, (self.new_width_l, self.new_height_l)
            )
            base_img_transformed = np.zeros((self.new_height_l, self.new_width_l, 3), dtype=np.uint8)
            base_img_transformed[
                self.corr_l[1] : self.corr_l[1] + base_img.shape[0],
                self.corr_l[0] : self.corr_l[0] + base_img.shape[1]
            ] = base_img

            stitched_img = cv2.bitwise_or(
                sec_img_transformed, cv2.bitwise_and(
                    base_img_transformed, cv2.bitwise_not(sec_img_transformed_mask)
                )
            )

            return stitched_img

        elif side == 'right':
            # Placing the images upon one another
            sec_img_transformed = cv2.warpPerspective(
                sec_img_cyl, self.H_r_corr, (self.new_width_r, self.new_height_r)
            )
            sec_img_transformed_mask = cv2.warpPerspective(
                sec_img_mask, self.H_r_corr, (self.new_width_r, self.new_height_r)
            )
            base_img_transformed = np.zeros((self.new_height_r, self.new_width_r, 3), dtype=np.uint8)
            base_img_transformed[
                self.corr_r[1] : self.corr_r[1] + base_img.shape[0],
                self.corr_r[0] : self.corr_r[0] + base_img.shape[1]
            ] = base_img

            stitched_img = cv2.bitwise_or(
                sec_img_transformed, cv2.bitwise_and(
                    base_img_transformed, cv2.bitwise_not(sec_img_transformed_mask)
                )
            )

            return stitched_img
        
        else:
            print("Invalid side value. Valid values: 'left', 'right'")


    def crop_borders(self):
        pass


    def blend_images(self):
        pass


    def create_panorama(self, images):
        """Main function."""

        # take out min(ii_x) and calculations with it into constructor
        base_img = self.project_onto_cylinder_preset(images[0])

        # stitch left image (pass homography and other args if needed)
        stitched_left = self.stitch_images(base_img, images[1], 'left')

        # stitch right image (pass stuff as above)
        pano = self.stitch_images(stitched_left, images[2], 'right')

        #pano = self.crop_borders(pano)

        return pano


if __name__ == '__main__':
    stitcher = CylindricalStitcher(720, 1280, f=1270)

    # Reading images.
    Images = stitcher.read_images("imgs/room")
    print("[INFO] Images read!")

    start1 = time.process_time()
    pano = stitcher.create_panorama(Images)
    stop1 = time.process_time() - start1

    print('Time 1: ', stop1)
    #cv2.imwrite("pano.png", pano)