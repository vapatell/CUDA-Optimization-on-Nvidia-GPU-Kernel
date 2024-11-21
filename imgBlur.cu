#include "libwb/wb.h"
#include "my_timer.h"

#define wbCheck(stmt)							\
  do {									\
    cudaError_t err = stmt;						\
    if (err != cudaSuccess) {						\
      wbLog(ERROR, "Failed to run stmt ", #stmt);			\
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));	\
      return -1;							\
    }									\
  } while (0)

#define BLUR_SIZE 21
#define BLOCK_DIM 8

//#define ALPHA 1
//#define TILE_DIM 8
//(BLOCK_DIM + (2 * BLUR_SIZE))
// #define FILTER_SIZE 15
// #define BLOCK_SIZE 16
// #define AUGMENTED (BLOCK_SIZE + 2 * FILTER_SIZE)

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE

// __global__ void blurKernel(float *out, float *in, int width, int height) {
//     // Define the size of the shared memory tile
//     extern __shared__ float tile[];

//     // Calculate global row and column index for the thread
//     int Col = blockIdx.x * blockDim.x + threadIdx.x;
//     int Row = blockIdx.y * blockDim.y + threadIdx.y;

//     // Calculate local row and column index within the block
//     int localCol = threadIdx.x;
//     int localRow = threadIdx.y;

//     // Calculate the starting position for the shared memory tile
//     int sharedWidth = blockDim.x + 2 * BLUR_SIZE;
//     //int sharedHeight = blockDim.y + 2 * BLUR_SIZE;

//     // Shared memory tile access
//     float *sharedTile = tile;

//     // Global memory boundary check
//     if (Col < width && Row < height) {
//         // Copy data from global memory to shared memory (including border)
//         int sharedRow = localRow + BLUR_SIZE;
//         int sharedCol = localCol + BLUR_SIZE;

//         // Load the main pixel into the center of the shared memory tile
//         sharedTile[sharedRow * sharedWidth + sharedCol] = in[Row * width + Col];

//         // Load the halo (border) pixels
//         if (localRow < BLUR_SIZE) {
//             // Top halo
//             int globalRow = max(Row - BLUR_SIZE, 0);
//             sharedTile[(localRow) * sharedWidth + sharedCol] = in[globalRow * width + Col];
//             // Bottom halo
//             globalRow = min(Row + blockDim.y, height - 1);
//             sharedTile[(localRow + blockDim.y + BLUR_SIZE) * sharedWidth + sharedCol] = in[globalRow * width + Col];
//         }
//         if (localCol < BLUR_SIZE) {
//             // Left halo
//             int globalCol = max(Col - BLUR_SIZE, 0);
//             sharedTile[sharedRow * sharedWidth + localCol] = in[Row * width + globalCol];
//             // Right halo
//             globalCol = min(Col + blockDim.x, width - 1);
//             sharedTile[sharedRow * sharedWidth + (localCol + blockDim.x + BLUR_SIZE)] = in[Row * width + globalCol];
//         }

//         __syncthreads(); // Ensure all threads have loaded their data into shared memory

//         // Perform the blur operation
//         float pixVal = 0.0f;
//         int numPixels = 0;

//         for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
//             for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
//                 int sharedRowIdx = sharedRow + blurRow;
//                 int sharedColIdx = sharedCol + blurCol;

//                 pixVal += static_cast<double>(sharedTile[sharedRowIdx * sharedWidth + sharedColIdx]);
//                 numPixels++;
//             }
//         }

//         // Write the result back to global memory
//         out[Row * width + Col] = static_cast<float>(pixVal / numPixels);
//     }
// }


// __global__ void blurKernel(float *out, float *in, int width, int height) 
// {
//   __shared__ float ds_in[TILE_DIM][TILE_DIM];

//   // Orig Col/Row bounded within the 8x8 or 16x16 - subtracting 2*BLUR_SIZE ensures this behaviour
//   int Col = blockIdx.x * (blockDim.x - (2 * BLUR_SIZE)) + threadIdx.x;
//   int Row = blockIdx.y * (blockDim.y - (2 * BLUR_SIZE)) + threadIdx.y;

//   int ty = threadIdx.y;
//   int tx = threadIdx.x;
  
//   if (Col < width && Row < height) 
//   {
//     int tileRow = Row - BLUR_SIZE;
//     int tileCol = Col - BLUR_SIZE;

//     if(tileRow > -1 && tileRow < height && tileCol > -1 && tileCol < width) 
//     {
//       ds_in[ty][tx] = in[tileRow * width + tileCol];
//     }
//     else
//     {
//       ds_in[ty][tx] = 0;
//     }

//     __syncthreads();

//     //printf("tx: %d\n", tx);
//     //printf("ty: %d\n", ty);

//     float pixVal = 0; int pixels = 0;
//     // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
//     //if((tx >= BLUR_SIZE) && (tx < TILE_DIM-BLUR_SIZE) && (ty >= BLUR_SIZE) && (ty < TILE_DIM-BLUR_SIZE))
//     //{  
//       for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) 
//       {
//         for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) 
//         {
//           int curRow = ty + blurRow;
//           int curCol = tx + blurCol;
//           //printf("curCol: %d\n", curCol);
//           //printf("curRow: %d\n", curRow);
//           // Verify we have a valid image pixel
//           if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) 
//           {
//             pixVal += ds_in[curRow][curCol];
//             //printf("pixVal: %f\n", pixVal);
//             // printf("curRow * width + curCol: %d\n", (curRow * width + curCol));
//             // printf("in: %f\n", in[curRow * width + curCol]);
//             // Keep track of number of pixels in the accumulated total
//             pixels++;
//           }
//         }
//       }
//     //}
//     //printf("pixVal: %f\n", pixVal);
//     //printf("pixels: %d\n", pixels);
//     // Write our new average pixel value out
//     out[tileRow * width + tileCol] = (float)(pixVal / pixels);
//     //printf("Row * width + Col: %d\n", Row * width + Col);
//     //printf("out: %f\n", out[Row * width + Col]);
//   }
// }

__global__ void blurKernel(float *out, float *in, int width, int height) 
{
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (Col < width && Row < height) 
  {
    float pixVal = 0; int pixels = 0;

    // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
    for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) 
    {
      for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) 
      {
        int curRow = Row + blurRow;
        int curCol = Col + blurCol;
        // Verify we have a valid image pixel
        if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) 
        {
          pixVal += in[curRow * width + curCol];
          // Keep track of number of pixels in the accumulated total
          pixels++;
        }
      }
    }

    // Write our new average pixel value out
    out[Row * width + Col] = (float)(pixVal / pixels);
  }
}

///////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  wbImage_t goldImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *goldOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);

  char *goldImageFile = argv[2];
  goldImage = wbImport(goldImageFile);

  // The input image is in grayscale, so the number of channels is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Pinning memory
  cudaHostAlloc((void **) &hostInputImageData, imageWidth*imageHeight* sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void **) &hostOutputImageData, imageWidth*imageHeight* sizeof(float), cudaHostAllocDefault);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
  //hostInputImageData  = wbImage_getData(inputImage);
  //hostOutputImageData = wbImage_getData(outputImage);
  goldOutputImageData = wbImage_getData(goldImage);

  // Copy data from the original host memory (wbImage_getData) to pinned memory
  float *tempInputImageData = wbImage_getData(inputImage);
  memcpy(hostInputImageData, tempInputImageData, imageWidth * imageHeight * sizeof(float));

  // Ensure `hostOutputImageData` is initialized if needed
  memset(hostOutputImageData, 0, imageWidth * imageHeight * sizeof(float));

  // Start timer
  timespec timer = tic();
  
  ////////////////////////////////////////////////
  //@@ INSERT AND UPDATE YOUR CODE HERE

  // Allocate cuda memory for device input and ouput image data
  cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 dimGrid((unsigned int)ceil(imageWidth / BLOCK_DIM), (unsigned int)ceil(imageHeight / BLOCK_DIM), 1);
  
  //size_t sharedMemSize = TILE_DIM * TILE_DIM * sizeof(float);
    
  // Call your GPU kernel 10 times
  for(int i = 0; i < 10; i++)
  {
    //blurKernel<<<dimGrid, dimBlock, sharedMemSize>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight);
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight);
  }
  
  // Transfer data from GPU to CPU
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
  ///////////////////////////////////////////////////////
  
  // Stop and print timer
  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // Check the correctness of your solution
  //wbSolution(args, outputImage);

   for(int i=0; i<imageHeight; i++){
     for(int j=0; j<imageWidth; j++){
       if(abs(hostOutputImageData[i*imageWidth+j]-goldOutputImageData[i*imageWidth+j])/goldOutputImageData[i*imageWidth+j]>0.01){
          printf("Incorrect output image at pixel (%d, %d): goldOutputImage = %f, hostOutputImage = %f\n", i, j, goldOutputImageData[i*imageWidth+j],hostOutputImageData[i*imageWidth+j]);
	        return -1;
       }
     }
   }
   printf("Correct output image!\n");

  cudaFreeHost(hostInputImageData);
  cudaFreeHost(hostOutputImageData);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
