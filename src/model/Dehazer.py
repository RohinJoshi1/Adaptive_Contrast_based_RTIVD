import cv2
import numpy as np
import argparse
import math
import time
import numpy as np

CLIP = lambda x: np.uint8(max(0, min(x, 255)))
AtmosphericLight_Y = 0
AtmosphericLight = np.zeros(3)

class Dehazing:
    def __init__(self, img_input):
        self.img_input = img_input
        self.imgY = cv2.cvtColor(img_input, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        self.AtmosphericLight = AtmosphericLight
        self.AtmosphericLight_Y = AtmosphericLight_Y
        self.width = self.img_input.shape[1]
        self.height = self.img_input.shape[0]
        self.pfTransmission = np.zeros(img_input.shape[:2])


    def GaussianTransmissionRefine(self):
            r = 29  # radius of the Gaussian filter

            # Apply Gaussian filtering to the transmission map
            t = cv2.GaussianBlur(self.pfTransmission, (r, r), 0)
            self.pfTransmission = t

    def AirLightEstimation(self, origin, height, width):
        UpperLeft  = self.img_input[origin[0]:origin[0]+int(round(height/2)), origin[1]:origin[1]+int(round(width/2))]
        UpperRight = self.img_input[origin[0]:origin[0]+int(round(height/2)), origin[1]+int(round(width/2)):origin[1]+width]
        LowerLeft  = self.img_input[origin[0]+int(round(height/2)):origin[0]+height, origin[1]:origin[1]+int(round(width/2))]
        LowerRight = self.img_input[origin[0]+int(round(height/2)):origin[0]+height, origin[1]+int(round(width/2)):origin[1]+width]

        if height*width > 200:
            maxVal = 0
            idx = -1
            for i, blk in enumerate([UpperLeft, UpperRight, LowerLeft, LowerRight]):
                D = np.mean(blk) - np.std(blk)
                if D > maxVal:
                    maxVal = D
                    idx = i
            self.AirLightEstimation(( origin[0]+int(idx/2)*int(round(height/2)),
                                      origin[1]+int(idx%2)*int(round(width/2))),
                                      int(round(height/2)), int(round(width/2)))
        else:
            global AtmosphericLight, AtmosphericLight_Y
            minDist = 1e10
            for i in range(height):
                for j in range(width):
                    Dist = np.linalg.norm(self.img_input[origin[0]+i,origin[1]+j,:] - np.array([255,255,255]))
                    if Dist < minDist:
                        minDist = Dist
                        self.AtmosphericLight = self.img_input[origin[0]+i, origin[1]+j,:]
                        ## RGB -> Y
                        self.AtmosphericLight_Y = int((self.AtmosphericLight[2]*0.299 + self.AtmosphericLight[1]*0.587 + self.AtmosphericLight[0]*0.114))
                        AtmosphericLight = self.AtmosphericLight
                        AtmosphericLight_Y = self.AtmosphericLight_Y

            ## renew airlight when abrupt change
            if abs(self.AtmosphericLight_Y - AtmosphericLight_Y) > 50:
                AtmosphericLight_Y = self.AtmosphericLight_Y
                AtmosphericLight = self.AtmosphericLight


    def TransmissionEstimation(self, blk_size):
        maxx = int((self.height // blk_size) * blk_size)
        maxy = int((self.width // blk_size) * blk_size)
        lamdaL = 4
        MinE = np.full(self.imgY.shape, 1e10)
        fOptTrs = np.zeros(self.imgY.shape)
        average = np.zeros(self.imgY.shape)

        for i in range(0, maxx, blk_size):
            for j in range(0, maxy, blk_size):
                average[i:i+blk_size,j:j+blk_size] = self.imgY[i:i+blk_size,j:j+blk_size].mean()

        for t, fTrans in enumerate(np.linspace(0.3,1,8)):
            over255 = np.zeros(self.imgY.shape)
            lower0 = np.zeros(self.imgY.shape)
            transed = (self.imgY.astype(int) - AtmosphericLight_Y)/fTrans + AtmosphericLight_Y

            Econtrast = -(transed - average)**2 / blk_size**2
            over255[transed > 255] = (transed[transed > 255] - 255)**2
            lower0[transed < 0] = (transed[transed < 0])**2
            for i in range(0, maxx, blk_size):
                for j in range(0, maxy, blk_size):
                    start = time.time()
                    E = Econtrast[i:i+blk_size,j:j+blk_size].sum() + lamdaL*(over255[i:i+blk_size,j:j+blk_size].sum() + lower0[i:i+blk_size,j:j+blk_size].sum())
                    if E < MinE[i][j]:
                        MinE[i:i+blk_size,j:j+blk_size] = E
                        fOptTrs[i:i+blk_size,j:j+blk_size] = fTrans
        self.pfTransmission = fOptTrs

    def RestoreImage(self,gamma=1.05):
        img_out = np.zeros(self.img_input.shape)
        self.pfTransmission = np.maximum(self.pfTransmission, np.full((self.height, self.width), 0.5))
        for i in range(3):
            img_out[:,:,i] = np.clip(((self.img_input[:,:,i].astype(int) - AtmosphericLight[i]) / self.pfTransmission + AtmosphericLight[i]),0,255)

        return img_out

    def GuidedFilter(self, rads, eps):
        ## Gudance image is ImgY , input image is pfTransmission
        meanI = cv2.boxFilter(self.imgY/255, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        meanP = cv2.boxFilter(self.pfTransmission, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        meanIP = cv2.boxFilter(self.imgY/255*self.pfTransmission, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        covIP = meanIP - meanI * meanP
        meanII = cv2.boxFilter((self.imgY/255)**2, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        varI = meanII - meanI ** 2
        a = covIP / (varI + eps)
        b = meanP - a * meanI
        meanA = cv2.boxFilter(a, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        meanB = cv2.boxFilter(b, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        res = meanA * self.imgY/255 + meanB
        self.pfTransmission = res
        self.pfTransmission = np.maximum(self.pfTransmission, np.full((self.height, self.width), 0.5))  # clip transmission => larger than 0.3

    def boxfilter(self,imSrc, r):
        """
        Performs O(1) time box filtering using cumulative sums.

        Args:
            imSrc (numpy.ndarray): The input image as a 2D NumPy array.
            r (int): The radius of the box filter (half-width of the filter window).

        Returns:
            numpy.ndarray: The filtered image using the box filter.
        """

        hei, wid = imSrc.shape  # Get image height and width

        # Initialize output image with zeros
        imDst = np.zeros_like(imSrc)

        # Cumulative sum over Y axis (optimized for clarity and potential speed benefits)
        imCumY = np.cumsum(imSrc, axis=0)  # Efficient cumulative sum using axis=0
        start_y, end_y = 1 + r, 1+2 * r   # Optimized window indices for Y cumulative sum differences
        imDst[:start_y, :] = imCumY[start_y:end_y, :]  # Efficient assignment for top boundary
        imDst[start_y:hei - r, :] = imCumY[end_y:, :] - imCumY[:-hei + 2 * r, :]  # Efficient assignment for middle rows
        imDst[hei - r:, :] = np.repeat(imCumY[-1, :], r)[np.newaxis, :] - imCumY[hei - 2 * r:, :]  # Efficient assignment for bottom boundary with broadcasting

        # Cumulative sum over X axis
        imCumX = np.cumsum(imDst, axis=1)  # Efficient cumulative sum using axis=1
        start_x, end_x = 1 + r, 2 * r + 1  # Optimized window indices for X cumulative sum differences
        imDst[:, :start_x] = imCumX[:, start_x:end_x]  # Efficient assignment for left boundary
        imDst[:, start_x:wid - r] = imCumX[:, end_x:] - imCumX[:, :-wid + 2 * r, :]  # Efficient assignment for middle columns
        imDst[:, wid - r:] = np.repeat(imCumX[:, -1], r)[:, np.newaxis] - imCumX[:, wid - 2 * r:, :]  # Efficient assignment for right boundary with broadcasting

        return imDst

    # def fast_gradient(self,r=20,eps=0.001) -> np.ndarray:
    #     """
    #     GUIDEDFILTER   O(1) time implementation of guided filter.

    #     - guidance image: I (should be a gray-scale/single channel image)
    #     - filtering input image: p (should be a gray-scale/single channel image)
    #     - regularization parameter: eps
    #     """
    #     height = 36
    #     width = 64
    #     im = self.imgY
    #     p = self.pfTransmission

    #     eps = eps ** 2
    #     N = self.boxfilter(np.ones((height,width), dtype=np.float32), r)
    #     N1 = self.boxfilter(np.ones((height,width), dtype=np.float32), 1)
    #     s_end = tuple(reversed(im.shape[:2]))
    #     s_start = (width, height)

    #     im_sub = cv2.resize(im, s_start, interpolation=cv2.INTER_NEAREST)
    #     p_sub = cv2.resize(p, s_start, interpolation=cv2.INTER_NEAREST)

    #     mean_im = np.divide(self.boxfilter(im_sub,r), N)
    #     mean_p = np.divide(self.boxfilter(p_sub, r), N)
    #     mean_imp = np.divide(self.boxfilter(np.multiply(im_sub, p_sub),r), N)
    #     # Covariance matrix of (im, p) in each local patch
    #     cov_imp = mean_imp - np.multiply(mean_im, mean_p)
    #     mean_imim = np.divide(self.boxfilter(np.multiply(im_sub, im_sub), r), N)
    #     var_im = mean_imim - np.multiply(mean_im, mean_im)
    #     r1 = 4
    #     # Weight
    #     epsilon = (0.01 * (np.max(p_sub) - np.min(p_sub)))**2

    #     # N1 = boxfilter(np.ones((args.heigt, args.width), dtype=np.float32), args.r1);
    #     # the size of each local patch; N=(2r+1)^2 except for boundary pixels.

    #     mean_im1 = np.divide(self.boxfilter(im_sub, r1), N1)
    #     mean_imim1 = np.divide(self.boxfilter(np.multiply(im_sub,im_sub),r1), N1)
    #     var_im1 = mean_imim1 - np.multiply(mean_im1, mean_im1)

    #     chi_im = np.sqrt(np.abs(var_im1, var_im))
    #     weight = (chi_im + epsilon) / (np.mean(chi_im) + epsilon)

    #     gamma = (4/np.mean(chi_im) - np.min(chi_im)) * (chi_im - np.mean(chi_im))
    #     gamma = 1 - np.divide(1, (1 + np.exp(gamma)))

    #     # Result
    #     a = (cov_imp + np.divide(np.multiply(np.divide(eps, weight), gamma), (var_im + np.divide(eps, weight))))
    #     b = mean_p - np.multiply(a, mean_im)

    #     mean_a = np.divide(self.boxfilter(a, r), N)
    #     mean_b = np.divide(self.boxfilter(b, r), N)
    #     mean_a = cv2.resize(mean_a, s_end, interpolation=cv2.INTER_LINEAR)
    #     mean_b = cv2.resize(mean_b, s_end, interpolation=cv2.INTER_LINEAR)

    #     q = np.multiply(mean_a, im) + mean_b
    #     return q


def main():
    im_name = "77.png"
    im = cv2.imread(f"/Users/rohinjoshi/Work/codes/MajorProject/playground/test/RTVD/final/input/{im_name}")
    # scale_factor = 0.5
    # dhz_img = im
    # # Get the dimensions of the original image
    # height, width = dhz_img.shape[:2]

    # # Calculate the new dimensions
    # new_height = int(height * scale_factor)
    # new_width = int(width * scale_factor)

    # # Downscale the image
    # downscaled_img = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # im = downscaled_img


    dehaze_img = Dehazing(im)
    start = time.time()
    dehaze_img.AirLightEstimation((0,0), im.shape[0], im.shape[1])
    print(dehaze_img.AtmosphericLight_Y)
    blk_size = 16
    dehaze_img.TransmissionEstimation(blk_size)
    print(dehaze_img.pfTransmission)
    dehaze_img.GaussianTransmissionRefine()
    eps = 0.001
    start = time.time()
    dehaze_img.GuidedFilter(20, eps)
    # print(f"GF: {(time.time()-start)*1000}ms")
    result_img = dehaze_img.RestoreImage().astype('uint8')
    # print(time.time()-start)
    # cv2.namedWindow('input_img', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('result_img', cv2.WINDOW_NORMAL)
    # cv2.imshow('input_img', im)
    # start = time.time()
    # cv2.imshow('result_img', result_img)
    # print(f"Cv2 takes {(time.time()-start)*1000}ms ")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f"/Users/rohinjoshi/Work/codes/MajorProject/playground/test/RTVD/final/output/{im_name}", result_img)


    # elif args.type == 'video':
    # video_capture = cv2.VideoCapture("/Users/rohinjoshi/Work/codes/MajorProject/playground/test/RTVD/hazyVideo.mp4")
    # ret, init = video_capture.read()
    # h, w = init.shape[0], init.shape[1]
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    # print("fps:", fps, ", width:", w, ", height:", h)

        # video_capture = cv2.VideoCapture("/Users/rohinjoshi/Work/codes/MajorProject/playground/test/RTVD/hazyVideo.mp4")
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter("./output_video.mp4", fourcc, fps, (w, h))
        # cnt = 0

    # while True:
    #     ret, frame = video_capture.read()
    #     cv2.namedWindow('input_img', cv2.WINDOW_NORMAL)
    #     cv2.imshow('input_img', frame)

    #     if ret == True and cnt % 2 == 0:                   # process every 2 frames -> avoid lag
    #         dehaze_img = Dehazing(frame)
    #         if cnt == 0:                                   # use the airlight of the first frame
    #             dehaze_img.AirLightEstimation((0,0), frame.shape[0], frame.shape[1])

    #         blk_size = 8
    #         dehaze_img.TransmissionEstimation(blk_size)
    #         eps = 0.001
    #         dehaze_img.GuidedFilter(20, eps)
    #         result_img = dehaze_img.RestoreImage().astype('uint8')
    #         cv2.namedWindow('result_img', cv2.WINDOW_NORMAL)
    #         cv2.imshow('result_img', result_img)
    #         out.write(result_img)
    #         #print(cnt)
    #     elif ret != True:
    #         video_capture.release()
    #         out.release()
    #         cv2.destroyAllWindows()
    #         break
    #     cnt += 1

    #     if cv2.waitKey(1) & 0xFF == ord('q'):   break

if __name__ == '__main__':
    main()
