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

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE
__global__ void blurKernel(float *out, float *in, int width, int height) 
{
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int BLUR_SIZE = 2;
  
  if (Col < w && Row < h) 
  {
    int pixVal = 0; int pixels = 0;
    // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
    for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) 
    {
      for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) 
      {
        int curRow = Row + blurRow;
        int curCol = Col + blurCol;
        // Verify we have a valid image pixel
        if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) 
        {
          pixVal += in[curRow * w + curCol];
          // Keep track of number of pixels in the accumulated total
          pixels++;
        }
      }
    }
    // Write our new average pixel value out
    out[Row * w + Col] = (unsigned char)(pixVal / pixels);
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

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  goldOutputImageData = wbImage_getData(goldImage);

  // Start timer
  timespec timer = tic();
  
  ////////////////////////////////////////////////
  //@@ INSERT AND UPDATE YOUR CODE HERE

  // Allocate cuda memory for device input and ouput image data
  cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(ceil(imgWidth/16.0), ceil(imgHeight/16.0), 1);
  
  // Call your GPU kernel 10 times
  for(int i = 0; i < 10; i++)
  imgBlurGPU<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight);

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
