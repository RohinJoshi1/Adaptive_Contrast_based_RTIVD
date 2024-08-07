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

    def calculate_fast_transmission_clip_metric(self,img_input, AtmosphericLight_Y):
    # Sample a small subset of pixels for speed
      sample_size = 50
      sampled_pixels = img_input.reshape(-1, 3)[np.random.choice(img_input.size // 3, sample_size, replace=False)]
      avg_intensity = np.mean(sampled_pixels)
      # Calculate a simple metric based on the difference between average intensity and AtmosphericLight_Y
      diff = abs(avg_intensity - AtmosphericLight_Y) / 255
      print(diff)
      metric = 0.3 + diff*0.8
      print(metric)
      return np.ceil(metric*10)/10



    def RestoreImage(self):
        img_out = np.zeros(self.img_input.shape)
        transmission_clip = self.calculate_fast_transmission_clip_metric(self.img_input, self.AtmosphericLight_Y)
        
        self.pfTransmission = np.maximum(self.pfTransmission, transmission_clip)
        
        for i in range(3):
            img_out[:,:,i] = np.clip(((self.img_input[:,:,i].astype(int) - self.AtmosphericLight[i]) / self.pfTransmission + self.AtmosphericLight[i]), 0, 255)
        
        return img_out

class FastDehazerGPU:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.prev_frame = None
        self.temporal_coherence_threshold = 0.8

        # CUDA kernel for motion vector calculation
        self.motion_vector_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void motion_vector_kernel(const float* frame1, const float* frame2, float2* motion_vectors, 
                                  int width, int height, int block_size) {
            int bx = blockIdx.x * block_size;
            int by = blockIdx.y * block_size;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            int x = bx + tx;
            int y = by + ty;
            
            if (x < width && y < height) {
                float diff = 0.0f;
                float edge_diff = 0.0f;
                
                for (int i = 0; i < block_size; i++) {
                    for (int j = 0; j < block_size; j++) {
                        int idx = (y + i) * width + (x + j);
                        if ((y + i) < height && (x + j) < width) {
                            diff += frame2[idx] - frame1[idx];
                            
                            // Simple edge detection
                            if (i > 0 && j > 0) {
                                float e1 = fabsf(frame1[idx] - frame1[idx - 1]) + fabsf(frame1[idx] - frame1[idx - width]);
                                float e2 = fabsf(frame2[idx] - frame2[idx - 1]) + fabsf(frame2[idx] - frame2[idx - width]);
                                edge_diff += fabsf(e2 - e1);
                            }
                        }
                    }
                }
                
                int block_idx = (y / block_size) * (width / block_size) + (x / block_size);
                motion_vectors[block_idx] = make_float2(diff / (block_size * block_size), edge_diff / (block_size * block_size));
            }
        }
        ''', 'motion_vector_kernel')


    def calculate_temporal_coherence(self, curr_frame):
        # Ensure curr_frame is on GPU
        if not isinstance(curr_frame, cp.ndarray):
            curr_frame = cp.asarray(curr_frame)
        
        # Convert to grayscale if necessary
        if curr_frame.ndim == 3:
            curr_frame = cp.mean(curr_frame, axis=2, dtype=cp.float32)
        else:
            curr_frame = curr_frame.astype(cp.float32)
        
        if self.prev_frame is None:
            self.prev_frame = curr_frame
            return 0.0
        
        height, width = curr_frame.shape
        block_size = self.block_size
        grid = ((width + block_size - 1) // block_size, (height + block_size - 1) // block_size)
        block = (block_size, block_size)
        
        motion_vectors = cp.empty((height // block_size, width // block_size, 2), dtype=cp.float32)
        
        self.motion_vector_kernel(grid, block, (self.prev_frame, curr_frame, motion_vectors, width, height, block_size))
        
        mean_motion = cp.mean(motion_vectors, axis=(0, 1))
        magnitude = cp.linalg.norm(mean_motion)
        coherence = cp.clip(10* cp.exp(-magnitude),0,1)  
        self.prev_frame = curr_frame
        return float(coherence)

    def set_temporal_coherence_threshold(self, threshold):
        self.temporal_coherence_threshold = threshold

    def get_temporal_coherence_threshold(self):
        return self.temporal_coherence_threshold

class FastDehazer(Dehazer):
    def __init__(self, img_input):
        super().__init__(img_input)
        self.prev_transmission = None
        self.gpu_dehazer = FastDehazerGPU(block_size=8) 
        self.gpu_dehazer.set_temporal_coherence_threshold(0.8)
        self.gpu_dehazer.block_size = 8

    def calculate_temporal_coherence(self, curr_frame):
        return self.gpu_dehazer.calculate_temporal_coherence(curr_frame)

    def process_frame(self, frame):
        tc = self.calculate_temporal_coherence(frame)
        if tc > self.gpu_dehazer.get_temporal_coherence_threshold() and self.prev_transmission is not None:
            print("TEMPORAL COHERENCE:",tc)
            self.pfTransmission = self.prev_transmission.copy()
            self.GuidedFilter_GPU(20, 0.01)
            

        else:
            self.AirLightEstimation((0,0), self.height, self.width)
            self.TransmissionEstimation(8)
            self.GaussianTransmissionRefine()
            self.GuidedFilter_GPU(20, 0.01)
        dehazed_frame = self.RestoreImage()
        self.prev_transmission = self.pfTransmission.copy()
        return dehazed_frame


def downscale_frame(dhz_img):
    scale_factor = 0.3
    #TODO dynamically scale image based on size : hardcoded to 300*400
    #OG img dim
    height, width = dhz_img.shape[:2]

    #new dim
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    # Downscale
    downscaled_img = cv2.resize(dhz_img, (400,300), interpolation=cv2.INTER_LINEAR)
    dhz_img = downscaled_img
    return dhz_img

def dehaze_img(img):
  dhz_img = downscale_frame(img)
  # dhz_img = img
  dhz = Dehazer(dhz_img)
  dhz.AirLightEstimation((0,0),dhz_img.shape[0],dhz_img.shape[1])
  blk_size = 8
  dhz.TransmissionEstimation(blk_size)

  dhz.GaussianTransmissionRefine()
  eps = 0.001
  dhz.GuidedFilter_GPU(20,eps)
  im = dhz.RestoreImage().astype('uint8')
  return im

def dehaze_video(video_url):
    video_capture = cv2.VideoCapture(video_url)
    ret, init = video_capture.read()
    h, w = init.shape[0], init.shape[1]
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print("fps:", fps, ", width:", w, ", height:", h)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("./output_video.mp4", fourcc, fps, (w, h))
    cnt = 0
    dehazer = None
    while True:
        ret, frame = video_capture.read()
        frame =  downscale_frame(frame)
        if dehazer is None: 
          dehazer = FastDehazer(frame)
        dehazed_frame = dehazer.process_frame(frame)
        cv2.imshow('Dehazed Frame',dehazed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
        # if ret == True and cnt % 2 == 0:
        #     dhz = Dehazer(frame)
        #     if cnt==0:                                   # use the airlight of the first frame
        #         dhz.AirLightEstimation((0,0), frame.shape[0], frame.shape[1])
        #     blk_size = 8
        #     dhz.TransmissionEstimation(blk_size)
        #     dhz.GaussianTransmissionRefine()
        #     eps = 0.001
        #     dhz.GuidedFilter_GPU(20,eps)
        #     im = dhz.RestoreImage().astype('uint8')
        #     cv2.namedWindow('result_img', cv2.WINDOW_NORMAL)
        #     cv2.imshow('result_img', im)
        #     out.write(im)
    #         #print(cnt)
        elif ret != True:
            video_capture.release()
            out.release()
            cv2.destroyAllWindows()
            break
        cnt += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):   break 



if __name__ == '__main__':
    # dehaze_img(cv2.imread('./test_images/test_image_1.jpg'))
    dehaze_video('./test_videos/test_video_1.mp4')
