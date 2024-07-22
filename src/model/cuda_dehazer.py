import cupyx.scipy.ndimage as cupy_filters

from numba import int64
import numba
from numba import cuda
import cupy as cp
import numpy as np
import time
CLIP = lambda x: np.uint8(max(0, min(x, 255)))
AtmosphericLight_Y = 0
AtmosphericLight = np.zeros(3)

## 30 FPS @ img size (300 x 400)[h,w]
@cuda.jit
def box_filter_kernel(input_arr, output_arr, width, height, radius):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    gw = cuda.gridDim.x
    gh = cuda.gridDim.y

    x, y = tx + bw * cuda.blockIdx.x, ty + bh * cuda.blockIdx.y

    if x < width and y < height:
        pixel_sum = 0.0
        num_pixels = 0

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                xx = x + i
                yy = y + j

                if xx >= 0 and xx < width and yy >= 0 and yy < height:
                    pixel_sum += input_arr[yy, xx]
                    num_pixels += 1

        output_arr[y, x] = pixel_sum / num_pixels


@cuda.jit(device=True)
def cached_trans(imgY_val, AtmosphericLight_Y, fTrans):
    return (imgY_val - AtmosphericLight_Y) / fTrans + AtmosphericLight_Y


class Dehazer:
    def __init__(self, img_input):
        self.img_input = img_input
        self.imgY = cv2.cvtColor(img_input, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        self.imgY_gpu = cp.asarray(self.imgY)
        self.AtmosphericLight_Y_gpu = cp.asarray(AtmosphericLight_Y)
        self.AtmosphericLight = AtmosphericLight
        self.AtmosphericLight_Y = AtmosphericLight_Y
        self.width = self.img_input.shape[1]
        self.height = self.img_input.shape[0]
        self.pfTransmission = np.zeros(img_input.shape[:2])

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


    def GaussianTransmissionRefine(self):
            r = 29  # radius of the Gaussian filter

            # Apply Gaussian filtering to the transmission map
            t = cv2.GaussianBlur(self.pfTransmission, (r, r), 0)
            # t = np.round(t,1)
            # t= np.ceil(t * 10) / 10
            self.pfTransmission = t


    @staticmethod
    @cuda.jit
    def calculate_average_kernel(imgY, average, blk_size, height, width):

        i, j = cuda.grid(2)
        bx, by = cuda.blockIdx.x, cuda.blockIdx.y
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        tile_width, tile_height = 8,8

        shared = cuda.shared.array(shape=(tile_height, tile_width), dtype='int64')

        for di in range(blk_size):
            for dj in range(blk_size):
                ii = i + di
                jj = j + dj
                if ii < height and jj < width:
                    shared[ty, tx] = imgY[ii, jj]

            cuda.syncthreads()

            sum_val = 0
            count = 0
            for dy in range(tile_height):
                for dx in range(tile_width):
                    ii = bx * tile_height + dy
                    jj = by * tile_width + dx
                    if ii < height and jj < width:
                        sum_val += shared[dy, dx]
                        count += 1
            average[i, j] = sum_val / count

    @staticmethod
    @cuda.jit
    def calculate_econtrast_kernel(imgY, average, fTrans, Econtrast, over255, lower0, AtmosphericLight_Y, blk_size, height, width):
        i, j = cuda.grid(2)
        bx, by = cuda.blockIdx.x, cuda.blockIdx.y
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        tile_width, tile_height = 8,8

        shared = cuda.shared.array(shape=(tile_height, tile_width), dtype='uint8')
        avg_shared = cuda.shared.array(shape=(tile_height,tile_width),dtype='uint8')

        for di in range(blk_size):
            for dj in range(blk_size):
                ii = i+di
                jj = j + dj
                if ii < height and jj < width:
                    shared[ty,tx] = imgY[ii,jj]
                    avg_shared[ty,tx] = average[ii,jj]

        cuda.syncthreads()

        # Calculate Econtrast, over255, and lower0 values for the tile
        if i < height and j < width:
            transed = cached_trans(shared[ty,tx], AtmosphericLight_Y, fTrans)
            diff = transed - avg_shared[ty,tx]
            Econtrast[i, j] = -diff ** 2 / blk_size ** 2
            over255[i, j] = ((transed > 255) * (transed - 255) ** 2)
            lower0[i, j] = ((transed < 0) * (transed) ** 2)



    @staticmethod
    @cuda.jit
    def update_min_e_kernel(Econtrast, over255, lower0, MinE, fOptTrs, lamdaL, height, width, fTrans):
        i, j = cuda.grid(2)
        if i < height and j < width:
            E = Econtrast[i, j] + lamdaL * (over255[i, j] + lower0[i, j])
            if E < MinE[i, j]:
                MinE[i, j] = E
                fOptTrs[i, j] = fTrans



    def TransmissionEstimation(self, blk_size):
        maxx = (self.height // blk_size) * blk_size
        maxy = (self.width // blk_size) * blk_size
        lamdaL = 4
        MinE_gpu = cp.full(self.imgY.shape, 1e10)
        fOptTrs_gpu = cp.zeros(self.imgY.shape)
        average_gpu = cp.zeros(self.imgY.shape)
        threads_per_block = (min(blk_size, self.height), min(blk_size, self.width))
        blocks_per_grid = ((maxx + threads_per_block[0]-1) // threads_per_block[0],
                          (maxy + threads_per_block[1]-1) // threads_per_block[1])
        start = time.time()
        self.calculate_average_kernel[blocks_per_grid, threads_per_block](self.imgY_gpu, average_gpu, blk_size, self.height, self.width)
        for t, fTrans in enumerate(np.linspace(0.3, 1, 8)):
            Econtrast_gpu = cp.zeros(self.imgY.shape)
            over255_gpu = cp.zeros(self.imgY.shape)
            lower0_gpu = cp.zeros(self.imgY.shape)
            start = time.time()
            #(imgY, average, fTrans, Econtrast, over255, lower0, AtmosphericLight_Y, blk_size, height, width)
            # (imgY, average, fTrans, Econtrast, over255, lower0, AtmosphericLight_Y, height, width):
            self.calculate_econtrast_kernel[blocks_per_grid, threads_per_block](self.imgY_gpu, average_gpu.astype('uint8'), fTrans, Econtrast_gpu, over255_gpu, lower0_gpu, self.AtmosphericLight_Y,blk_size, self.height, self.width)
            start = time.time()
            self.update_min_e_kernel[blocks_per_grid, threads_per_block](Econtrast_gpu, over255_gpu, lower0_gpu, MinE_gpu, fOptTrs_gpu, lamdaL, self.height, self.width, fTrans)
        self.pfTransmission = cp.asnumpy(fOptTrs_gpu)



    def box_filter(self,arr, radius):
      width, height = arr.shape
      output_arr = cp.zeros((height, width), dtype='float32')

      threads_per_block = (16, 16)
      blocks_per_grid_x = int(np.ceil(width / threads_per_block[0]))
      blocks_per_grid_y = int(np.ceil(height / threads_per_block[1]))
      blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

      box_filter_kernel[blocks_per_grid, threads_per_block](arr, output_arr, width, height, radius)

      return output_arr

    def GuidedFilter_GPU(self, rads, eps):
        self.pfTransmission_gpu = cp.asarray(self.pfTransmission)
        meanI = cupy_filters.uniform_filter(self.imgY_gpu / 255, size=(rads, rads), mode='reflect')
        meanP = cupy_filters.uniform_filter(self.pfTransmission_gpu, size=(rads, rads), mode='reflect')
        meanIP = cupy_filters.uniform_filter(self.imgY_gpu / 255 * self.pfTransmission_gpu, size=(rads, rads), mode='reflect')
        covIP = meanIP - meanI * meanP
        meanII = cupy_filters.uniform_filter((self.imgY_gpu / 255) ** 2, size=(rads, rads), mode='reflect')
        varI = meanII - meanI ** 2
        a = covIP / (varI + eps)
        b = meanP - a * meanI
        meanA = cupy_filters.uniform_filter(a, size=(rads, rads), mode='reflect')
        meanB = cupy_filters.uniform_filter(b, size=(rads, rads), mode='reflect')
        res = meanA * self.imgY_gpu / 255 + meanB
        self.pfTransmission_gpu = cp.maximum(res, cp.full((self.height, self.width), 0.3))


    def GuidedFilter_CPU(self, rads, eps):
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
        self.pfTransmission = np.maximum(self.pfTransmission, np.full((self.height, self.width), 0.3))  # clip transmission => larger than 0.3

    def calculate_fast_transmission_clip_metric(self,img_input, AtmosphericLight_Y):
    # Sample a small subset of pixels for speed
      sample_size = 50
      sampled_pixels = img_input.reshape(-1, 3)[np.random.choice(img_input.size // 3, sample_size, replace=False)]
      
      # Calculate the average intensity of the sampled pixels
      avg_intensity = np.mean(sampled_pixels)
      
      # Calculate a simple metric based on the difference between average intensity and AtmosphericLight_Y
      diff = abs(avg_intensity - AtmosphericLight_Y) / 255
      print(diff)
  
      # Map the difference to a range between 0.3 and 0.7
      metric = 0.3 + diff*0.8
      print(metric)
      return np.ceil(metric*10)/10



    def RestoreImage(self):
        img_out = np.zeros(self.img_input.shape)
        
        # Calculate the transmission clip value using our new ultra-fast metric
        transmission_clip = self.calculate_fast_transmission_clip_metric(self.img_input, self.AtmosphericLight_Y)
        
        self.pfTransmission = np.maximum(self.pfTransmission, transmission_clip)
        
        for i in range(3):
            img_out[:,:,i] = np.clip(((self.img_input[:,:,i].astype(int) - self.AtmosphericLight[i]) / self.pfTransmission + self.AtmosphericLight[i]), 0, 255)
        
        return img_out
