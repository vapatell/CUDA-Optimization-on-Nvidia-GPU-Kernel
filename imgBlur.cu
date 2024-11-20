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
//#define FILTER_SIZE 11

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE
__global__ void blurKernel(float *out, float *in, int width, int height) 
{
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  //printf("Col: %d\n", Col);
  //printf("Row: %d\n", Row);
  
  // for(int i = 0; i < 25; i++)
  // {
  //   printf("img[%d]: %f \n", i, in[i]);
  //   //printf("goldimg[%d]: %f \n", i, goldOutputImageData[i]);
  // }

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
        //printf("curCol: %d\n", curCol);
        //printf("curRow: %d\n", curRow);
        // Verify we have a valid image pixel
        if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) 
        {
          pixVal += in[curRow * width + curCol];
          // printf("pixVal: %f\n", pixVal);
          // printf("curRow * width + curCol: %d\n", (curRow * width + curCol));
          // printf("in: %f\n", in[curRow * width + curCol]);
          // Keep track of number of pixels in the accumulated total
          pixels++;
        }
      }
    }
    //printf("pixVal: %f\n", pixVal);
    //printf("pixels: %d\n", pixels);
    // Write our new average pixel value out
    out[Row * width + Col] = (float)(pixVal / pixels);
    //printf("Row * width + Col: %d\n", Row * width + Col);
    //printf("out: %f\n", out[Row * width + Col]);
  }
}

// __global__ void blurKernel(float *output, float *input, int width, int height)
// {
//     // calculate global row and column index
//     int Col = blockIdx.x * blockDim.x + threadIdx.x;
//     int Row = blockIdx.y * blockDim.y + threadIdx.y;

//     // ensure thread is processing a valid pixel within image boundaries
//     if(Col < width && Row < height){
//         int pixVal = 0; 
//         int pixels = 0;
//         int BLUR_SIZE = 3; // size of blur kernel (3x3)

//         // iterate over blur kernel
//         for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
//             for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE +1; ++blurCol){

//                 int curRow = Row + blurRow;
//                 int curCol = Col + blurCol;

//                 // check if neighbouring poxel is within image boundaries
//                 if (curRow > -1 && curRow < height && curCol > -1 && curCol < width){
//                     pixVal += input[curRow * width + curCol];
//                     pixels++;
//                 }
//             }
//         }

//         output[Row*width +Col] = (float) (pixVal / pixels); // calculate average of pixel value
//     }
// }

// __global__ void blurKernel(float* outImg, float* inImg, int width, int height) {
//     int filterRow, filterCol;
//     int cornerRow, cornerCol;
//     int tx = threadIdx.x; int ty = threadIdx.y;
//     int bx = blockIdx.x; int by = blockIdx.y;
//     int filterSize = 2*FILTER_SIZE + 1;

//     // compute global thread coordinates
//     int row = by * blockDim.y + ty;
//     int col = bx * blockDim.x + tx;

//     // make sure thread is within image boundaries
//     if ((row < height) && (col < width)) {
//         // instantiate accumulator
//         int numPixels = 0;
//         int cumSum = 0;

//         // top-left corner coordinates
//         cornerRow = row - FILTER_SIZE;
//         cornerCol = col - FILTER_SIZE;

//         // accumulate values inside filter
//         for (int i = 0; i < filterSize; i++) {
//             for (int j = 0; j < filterSize; j++) {
//                 // filter coordinates
//                 filterRow = cornerRow + i;
//                 filterCol = cornerCol + j;

//                 // accumulate sum
//                 if ((filterRow >= 0) && (filterRow <= height) && (filterCol >= 0) && (filterCol <= width)) {
//                     cumSum += inImg[filterRow*width + filterCol];
//                     numPixels++;
//                 }
//             }
//         }
//         // set the value of output
//         outImg[row*width + col] = (float)(cumSum / numPixels);
//     }
// }
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

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  goldOutputImageData = wbImage_getData(goldImage);

  printf("imgWidth: %d\n", imageWidth);
  printf("imgHeight: %d\n", imageHeight);

  // Start timer
  timespec timer = tic();
  
  ////////////////////////////////////////////////
  //@@ INSERT AND UPDATE YOUR CODE HERE

  // Allocate cuda memory for device input and ouput image data
  cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);
  
  // dim3 dimBlock(16, 16, 1);
  // dim3 dimGrid(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);

  dim3 dimBlock(1024, 1, 1);
  dim3 dimGrid(3, 3840, 1);
  
  // Call your GPU kernel 10 times
  for(int i = 0; i < 11; i++)
  {
    //printf("iter: %d\n", i);
    blurKernel<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight);
    //blurKernel<<<2, 1025>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight);
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

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
