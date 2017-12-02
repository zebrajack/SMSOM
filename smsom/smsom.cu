/*
 * This single source code is a straightforward implementation of our paper [1]. See http://zhaozj89.github.io/SMSOM/ for more details. 
 * Written by Zhenjie Zhao, if you have any question, please feel free to contact <zhaozj89@gmail.com>.
 * 
 * The executable smsom.exe in /SMSOM/Debug is built under the following environment:
 * 1. Visual Studio 2010
 * 2. CUDA 5.0
 * 3. OpenCV 2.4.5
 * 4. Windows 7 (64 bit)
 * You can use the executable directly in a similar environment. Alternatively, you can build it in other environments manually. See README.md file for more details.
 *
 * [1] Zhenjie Zhao, Xuebo Zhang, and Yongchun Fang. Stacked Multi-layer Self-Organizing Map for Background Modeling. IEEE Transactions on Image Processing, 2015, Accepted.
 */

// cuda5.0
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// std
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring>

//
using namespace cv;
using namespace std;

//
__device__ const float PI = 3.1415926;
__device__ float gaussKernel[3][3] = {1/16.0, 2/16.0, 1/16.0, 2/16.0, 4/16.0, 2/16.0, 1/16.0, 2/16.0, 1/16.0};

__device__ int mi[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
__device__ int mj[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};

__device__ int xlu[9] = {-1, 0, 0, -1, 0, 0, -1, 0, 0};
__device__ int xu[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ int xru[9] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
__device__ int xr[9] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
__device__ int xrd[9] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
__device__ int xd[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ int xld[9] = {-1, 0, 0, -1, 0, 0, -1, 0, 0};
__device__ int xl[9] = {-1, 0, 0, -1, 0, 0, -1, 0, 0};

__device__ int ylu[9] = {-1, -1, -1, 0, 0, 0, 0, 0, 0};
__device__ int yu[9] = {-1, -1, -1, 0, 0, 0, 0, 0, 0};
__device__ int yru[9] = {-1, -1, -1, 0, 0, 0, 0, 0, 0};
__device__ int yr[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ int yrd[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
__device__ int yd[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
__device__ int yld[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
__device__ int yl[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ bool shadowRemove(float hi, float si, float vi,
	float hm, float sm, float vm){
		return ( (vi/vm<1) && (vi/vm>0.7) && (si-sm<0.1) && (fabs(hi-hm)<10) );
}

__device__ float distance(float h1, float s1, float v1, 
	float h2, float s2, float v2){
		return sqrtf(pow(s1*v1*cos(h1*PI/180) - s2*v2*cos(h2*PI/180), 2) +
			pow(s1*v1*sin(h1*PI/180) - s2*v2*sin(h2*PI/180), 2) +
			pow(v1 - v2, 2));
}

//
__global__ void initLayer(float* input, float* output, int width){
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y; // thread index

	for (int j=0; j<3; ++j){
		for (int i=0; i<3; ++i){
			output[(y*3+j)*width*3+(x*3+i)] = input[y*width+x];
		}
	}
}

//
__global__ void compete(float* modelH, float* modelS, float* modelV, 
	float* frameH, float* frameS, float* frameV, 
	bool* match, int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = modelH[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = modelS[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = modelV[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int index = 0;
		int i2 = 0;
		float min = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist <= min){
				min = dist; 
				index = i2;
			}
		}

		for (int j3 = 0; j3 < 3; ++j3){
			for (int i3 = 0; i3 < 3; ++i3){
				match[(y*3+j3)*width*3+(x*3+i3)] = false;
			}
		}
		match[(y*3+mj[index])*width*3+(x*3+mi[index])] = true;
}

__global__ void competeWithFilter(float* model1H, float* model1S, float* model1V,
	float* model2H, float* model2S, float* model2V,
	float* frameH, float* frameS, float* frameV,
	float* maxValue,
	bool* match, int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = model1H[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = model1S[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = model1V[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int i2 = 0;
		float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist >= max)
				max = dist; 
		}

		for (int j3 = 0; j3 < 3; ++j3){
			for (int i3 = 0; i3 < 3; ++i3){
				match[(y*3+j3)*width*3+(x*3+i3)] = false;
			}
		}

		if( max >= maxValue[y*width+x] ){
			for (int j = 0; j < 3; ++j){
				for (int i = 0; i < 3; ++i){
					pointModel[j*3+i][0] = model2H[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][1] = model2S[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][2] = model2V[(y*3+j)*width*3+(x*3+i)];
				}
			}

			int index = 0;
			int i2 = 0;
			float min = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

			for (int i2 = 1; i2 < 3*3; ++i2){ 
				float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
					pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
				if (dist <= min){
					min = dist; 
					index = i2;
				}
			}
			match[(y*3+mj[index])*width*3+(x*3+mi[index])] = true;
		}
}

__global__ void competeWithFilter2(float* model1H, float* model1S, float* model1V,
	float* model2H, float* model2S, float* model2V,
	float* model3H, float* model3S, float* model3V,
	float* frameH, float* frameS, float* frameV,
	float* maxValue1,
	float* maxValue2,
	bool* match, int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = model1H[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = model1S[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = model1V[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int i2 = 0;
		float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist >= max)
				max = dist; 
		}

		for (int j3 = 0; j3 < 3; ++j3){
			for (int i3 = 0; i3 < 3; ++i3){
				match[(y*3+j3)*width*3+(x*3+i3)] = false;
			}
		}

		if( max >= maxValue1[y*width+x] ){
			for (int j = 0; j < 3; ++j){
				for (int i = 0; i < 3; ++i){
					pointModel[j*3+i][0] = model2H[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][1] = model2S[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][2] = model2V[(y*3+j)*width*3+(x*3+i)];
				}
			}

			int i2 = 0;
			float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

			for (int i2 = 1; i2 < 3*3; ++i2){ 
				float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
					pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
				if (dist >= max)
					max = dist; 
			}

			if( max >=maxValue2[y*width+x] ){
				for (int j = 0; j < 3; ++j){
					for (int i = 0; i < 3; ++i){
						pointModel[j*3+i][0] = model3H[(y*3+j)*width*3+(x*3+i)];
						pointModel[j*3+i][1] = model3S[(y*3+j)*width*3+(x*3+i)];
						pointModel[j*3+i][2] = model3V[(y*3+j)*width*3+(x*3+i)];
					}
				}

				int index = 0;
				int i2 = 0;
				float min = distance(pointFrame[0], pointFrame[1], pointFrame[2],
					pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

				for (int i2 = 1; i2 < 3*3; ++i2){ 
					float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
						pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
					if (dist <= min){
						min = dist; 
						index = i2;
					}
				}
				match[(y*3+mj[index])*width*3+(x*3+mi[index])] = true;
			} // if
		} // if
}

// update the background model
__global__ void cooperate(float* modelH, float* modelS, float* modelV, 
	float* backupH, float* backupS, float* backupV,
	float* frameH, float* frameS, float* frameV,
	bool* match, 
	int width, int height, float alpha){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	int m = 0;
	for(int j = 0; j < 3; ++j){
		for(int i = 0; i < 3; ++i){
			m = j*3+i;
			// center
			if(match[(y*3+j)*width*3+(x*3+i)] == true){
				modelH[(y*3+j)*width*3+(x*3+i)] = 
					(1-alpha*gaussKernel[1][1])*backupH[(y*3+j)*width*3+(x*3+i)]
				+ alpha*gaussKernel[1][1]*(frameH[y*width+x]);
				modelS[(y*3+j)*width*3+(x*3+i)] =
					(1-alpha*gaussKernel[1][1])*backupS[(y*3+j)*width*3+(x*3+i)] 
				+ alpha*gaussKernel[1][1]*(frameS[y*width+x]);
				modelV[(y*3+j)*width*3+(x*3+i)] =
					(1-alpha*gaussKernel[1][1])*backupV[(y*3+j)*width*3+(x*3+i)] + 
					alpha*gaussKernel[1][1]*(frameV[y*width+x]);
			}
			// left up
			if (  (x+xlu[m])>=0 && (y+ylu[m])>=0 && 
				match[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][2])*backupH[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] 
					+ alpha*gaussKernel[2][2]*(frameH[(y+ylu[m])*width+(x+xlu[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][2])*backupS[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
					+ alpha*gaussKernel[2][2]*(frameS[(y+ylu[m])*width+(x+xlu[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][2])*backupV[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
					+ alpha*gaussKernel[2][2]*(frameV[(y+ylu[m])*width+(x+xlu[m])]);
			}
			// up
			if (  (y+yu[m])>=0 && 
				match[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][1])*backupH[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
					+ alpha*gaussKernel[2][1]*(frameH[(y+yu[m])*width+(x+xu[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][1])*backupS[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
					+ alpha*gaussKernel[2][1]*(frameS[(y+yu[m])*width+(x+xu[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][1])*backupV[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
					+ alpha*gaussKernel[2][1]*(frameV[(y+yu[m])*width+(x+xu[m])]);
			}
			// right up
			if (  (x+xru[m])<=width && (y+yru[m])>=0 && 
				match[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][0])*backupH[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
					+ alpha*gaussKernel[2][0]*(frameH[(y+yru[m])*width+(x+xru[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] =
						(1-alpha*gaussKernel[2][0])*backupS[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] 
					+ alpha*gaussKernel[2][0]*(frameS[(y+yru[m])*width+(x+xru[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[2][0])*backupV[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
					+ alpha*gaussKernel[0][2]*(frameV[(y+yru[m])*width+(x+xru[m])]);
			}
			// right
			if (  (x+xr[m])<=width && 
				match[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][0])*backupH[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
					+ alpha*gaussKernel[1][0]*(frameH[(y+yr[m])*width+(x+xr[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][0])*backupS[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
					+ alpha*gaussKernel[1][0]*(frameS[(y+yr[m])*width+(x+xr[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][0])*backupV[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
					+ alpha*gaussKernel[1][0]*(frameV[(y+yr[m])*width+(x+xr[m])]);
			}
			// right down
			if (  (x+xrd[m])<=width && (y+yrd[m])>=height && 
				match[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i] ==true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][0])*backupH[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
					+ alpha*gaussKernel[0][0]*(frameH[(y+yrd[m])*width+(x+xrd[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][0])*backupS[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
					+ alpha*gaussKernel[0][0]*(frameS[(y+yrd[m])*width+(x+xrd[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][0])*backupV[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
					+ alpha*gaussKernel[0][0]*(frameV[(y+yrd[m])*width+(x+xrd[m])]);
			}
			// down
			if (  (y+yd[m])>=height && 
				match[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][1])*backupH[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
					+ alpha*gaussKernel[0][1]*(frameH[(y+yd[m])*width+(x+xd[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][1])*backupS[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
					+ alpha*gaussKernel[0][1]*(frameS[(y+yd[m])*width+(x+xd[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][1])*backupV[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
					+ alpha*gaussKernel[0][1]*(frameV[(y+yd[m])*width+(x+xd[m])]);
			}
			// left down7
			if (  (y+yld[m])>=height && (x+xld[m])>=0 && 
				match[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] =
						(1-alpha*gaussKernel[0][2])*backupH[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
					+ alpha*gaussKernel[0][2]*(frameH[(y+yld[m])*width+(x+xld[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][2])*backupS[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
					+ alpha*gaussKernel[0][2]*(frameS[(y+yld[m])*width+(x+xld[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[0][2])*backupV[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
					+ alpha*gaussKernel[0][2]*(frameV[(y+yld[m])*width+(x+xld[m])]);
			}
			// left
			if (  (x+xl[m])>=0 && 
				match[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i] == true  ){
					modelH[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][2])*backupH[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
					+ alpha*gaussKernel[1][2]*(frameH[(y+yl[m])*width+(x+xl[m])]);
					modelS[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][2])*backupS[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
					+ alpha*gaussKernel[1][2]*(frameS[(y+yl[m])*width+(x+xl[m])]);
					modelV[(y*3+j)*width*3+(x*3+i)] = 
						(1-alpha*gaussKernel[1][2])*backupV[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
					+ alpha*gaussKernel[1][2]*(frameV[(y+yl[m])*width+(x+xl[m])]);
			}
		}
	}
}

__global__ void cooperateWithFilter(float* model1H, float* model1S, float* model1V,
	float* model2H, float* model2S, float* model2V, 
	float* backup2H, float* backup2S, float* backup2V,
	float* frameH, float* frameS, float* frameV,
	float* maxValue,
	bool* match, 
	int width, int height, float alpha){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = model1H[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = model1S[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = model1V[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int i2 = 0;
		float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist >= max)
				max = dist; 
		}

		if( max >= maxValue[y*width+x] ){
			int m = 0;
			for(int j = 0; j < 3; ++j){
				for(int i = 0; i < 3; ++i){
					m = j*3+i;
					// center
					if(match[(y*3+j)*width*3+(x*3+i)] == true){
						model2H[(y*3+j)*width*3+(x*3+i)] = 
							(1-alpha*gaussKernel[1][1])*backup2H[(y*3+j)*width*3+(x*3+i)]
						+ alpha*gaussKernel[1][1]*(frameH[y*width+x]);
						model2S[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup2S[(y*3+j)*width*3+(x*3+i)] 
						+ alpha*gaussKernel[1][1]*(frameS[y*width+x]);
						model2V[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup2V[(y*3+j)*width*3+(x*3+i)] + 
							alpha*gaussKernel[1][1]*(frameV[y*width+x]);
					}
					// left up
					if (  (x+xlu[m])>=0 && (y+ylu[m])>=0 && 
						match[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2H[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] 
							+ alpha*gaussKernel[2][2]*(frameH[(y+ylu[m])*width+(x+xlu[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2S[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameS[(y+ylu[m])*width+(x+xlu[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2V[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameV[(y+ylu[m])*width+(x+xlu[m])]);
					}
					// up
					if (  (y+yu[m])>=0 && 
						match[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2H[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameH[(y+yu[m])*width+(x+xu[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2S[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameS[(y+yu[m])*width+(x+xu[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2V[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameV[(y+yu[m])*width+(x+xu[m])]);
					}
					// right up
					if (  (x+xru[m])<=width && (y+yru[m])>=0 && 
						match[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup2H[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[2][0]*(frameH[(y+yru[m])*width+(x+xru[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[2][0])*backup2S[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] 
							+ alpha*gaussKernel[2][0]*(frameS[(y+yru[m])*width+(x+xru[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup2V[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yru[m])*width+(x+xru[m])]);
					}
					// right
					if (  (x+xr[m])<=width && 
						match[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2H[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameH[(y+yr[m])*width+(x+xr[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2S[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameS[(y+yr[m])*width+(x+xr[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2V[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameV[(y+yr[m])*width+(x+xr[m])]);
					}
					// right down
					if (  (x+xrd[m])<=width && (y+yrd[m])>=height && 
						match[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i] ==true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2H[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameH[(y+yrd[m])*width+(x+xrd[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2S[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameS[(y+yrd[m])*width+(x+xrd[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2V[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameV[(y+yrd[m])*width+(x+xrd[m])]);
					}
					// down
					if (  (y+yd[m])>=height && 
						match[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2H[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameH[(y+yd[m])*width+(x+xd[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2S[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameS[(y+yd[m])*width+(x+xd[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2V[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameV[(y+yd[m])*width+(x+xd[m])]);
					}
					// left down7
					if (  (y+yld[m])>=height && (x+xld[m])>=0 && 
						match[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[0][2])*backup2H[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameH[(y+yld[m])*width+(x+xld[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup2S[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameS[(y+yld[m])*width+(x+xld[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup2V[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yld[m])*width+(x+xld[m])]);
					}
					// left
					if (  (x+xl[m])>=0 && 
						match[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2H[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameH[(y+yl[m])*width+(x+xl[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2S[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameS[(y+yl[m])*width+(x+xl[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2V[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameV[(y+yl[m])*width+(x+xl[m])]);
					}
				}
			}
		}
}

__global__ void cooperateWithFilter2(float* model1H, float* model1S, float* model1V,
	float* model2H, float* model2S, float* model2V,
	float* model3H, float* model3S, float* model3V,
	float* backup3H, float* backup3S, float* backup3V,
	float* frameH, float* frameS, float* frameV,
	float* maxValue1,
	float* maxValue2,
	bool* match, 
	int width, int height, float alpha){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = model1H[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = model1S[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = model1V[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int i2 = 0;
		float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist >= max)
				max = dist; 
		}

		if( max >= maxValue1[y*width+x] ){
			for (int j = 0; j < 3; ++j){
				for (int i = 0; i < 3; ++i){
					pointModel[j*3+i][0] = model2H[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][1] = model2S[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][2] = model2V[(y*3+j)*width*3+(x*3+i)];
				}
			}

			int i2 = 0;
			float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

			for (int i2 = 1; i2 < 3*3; ++i2){ 
				float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
					pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
				if (dist >= max)
					max = dist; 
			}

			if( max>= maxValue2[y*width+x] ){
				int m = 0;
				for(int j = 0; j < 3; ++j){
					for(int i = 0; i < 3; ++i){
						m = j*3+i;
						// center
						if(match[(y*3+j)*width*3+(x*3+i)] == true){
							model3H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][1])*backup3H[(y*3+j)*width*3+(x*3+i)]
							+ alpha*gaussKernel[1][1]*(frameH[y*width+x]);
							model3S[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[1][1])*backup3S[(y*3+j)*width*3+(x*3+i)] 
							+ alpha*gaussKernel[1][1]*(frameS[y*width+x]);
							model3V[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[1][1])*backup3V[(y*3+j)*width*3+(x*3+i)] + 
								alpha*gaussKernel[1][1]*(frameV[y*width+x]);
						}
						// left up
						if (  (x+xlu[m])>=0 && (y+ylu[m])>=0 && 
							match[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] == true  ){
								model3H[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[2][2])*backup3H[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] 
								+ alpha*gaussKernel[2][2]*(frameH[(y+ylu[m])*width+(x+xlu[m])]);
								model3S[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[2][2])*backup3S[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
								+ alpha*gaussKernel[2][2]*(frameS[(y+ylu[m])*width+(x+xlu[m])]);
								model3V[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[2][2])*backup3V[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
								+ alpha*gaussKernel[2][2]*(frameV[(y+ylu[m])*width+(x+xlu[m])]);
						}
						// up
						if (  (y+yu[m])>=0 && 
							match[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i] == true  ){
								model3H[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[2][1])*backup3H[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
								+ alpha*gaussKernel[2][1]*(frameH[(y+yu[m])*width+(x+xu[m])]);
								model3S[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[2][1])*backup3S[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
								+ alpha*gaussKernel[2][1]*(frameS[(y+yu[m])*width+(x+xu[m])]);
								model3V[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[2][1])*backup3V[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
								+ alpha*gaussKernel[2][1]*(frameV[(y+yu[m])*width+(x+xu[m])]);
						}
						// right up
						if (  (x+xru[m])<=width && (y+yru[m])>=0 && 
							match[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] == true  ){
								model3H[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[2][0])*backup3H[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
								+ alpha*gaussKernel[2][0]*(frameH[(y+yru[m])*width+(x+xru[m])]);
								model3S[(y*3+j)*width*3+(x*3+i)] =
									(1-alpha*gaussKernel[2][0])*backup3S[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] 
								+ alpha*gaussKernel[2][0]*(frameS[(y+yru[m])*width+(x+xru[m])]);
								model3V[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[2][0])*backup3V[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
								+ alpha*gaussKernel[0][2]*(frameV[(y+yru[m])*width+(x+xru[m])]);
						}
						// right
						if (  (x+xr[m])<=width && 
							match[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i] == true  ){
								model3H[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[1][0])*backup3H[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
								+ alpha*gaussKernel[1][0]*(frameH[(y+yr[m])*width+(x+xr[m])]);
								model3S[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[1][0])*backup3S[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
								+ alpha*gaussKernel[1][0]*(frameS[(y+yr[m])*width+(x+xr[m])]);
								model3V[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[1][0])*backup3V[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
								+ alpha*gaussKernel[1][0]*(frameV[(y+yr[m])*width+(x+xr[m])]);
						}
						// right down
						if (  (x+xrd[m])<=width && (y+yrd[m])>=height && 
							match[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i] ==true  ){
								model3H[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[0][0])*backup3H[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
								+ alpha*gaussKernel[0][0]*(frameH[(y+yrd[m])*width+(x+xrd[m])]);
								model3S[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[0][0])*backup3S[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
								+ alpha*gaussKernel[0][0]*(frameS[(y+yrd[m])*width+(x+xrd[m])]);
								model3V[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[0][0])*backup3V[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
								+ alpha*gaussKernel[0][0]*(frameV[(y+yrd[m])*width+(x+xrd[m])]);
						}
						// down
						if (  (y+yd[m])>=height && 
							match[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i] == true  ){
								model3H[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[0][1])*backup3H[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
								+ alpha*gaussKernel[0][1]*(frameH[(y+yd[m])*width+(x+xd[m])]);
								model3S[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[0][1])*backup3S[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
								+ alpha*gaussKernel[0][1]*(frameS[(y+yd[m])*width+(x+xd[m])]);
								model3V[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[0][1])*backup3V[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
								+ alpha*gaussKernel[0][1]*(frameV[(y+yd[m])*width+(x+xd[m])]);
						}
						// left down7
						if (  (y+yld[m])>=height && (x+xld[m])>=0 && 
							match[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i] == true  ){
								model3H[(y*3+j)*width*3+(x*3+i)] =
									(1-alpha*gaussKernel[0][2])*backup3H[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
								+ alpha*gaussKernel[0][2]*(frameH[(y+yld[m])*width+(x+xld[m])]);
								model3S[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[0][2])*backup3S[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
								+ alpha*gaussKernel[0][2]*(frameS[(y+yld[m])*width+(x+xld[m])]);
								model3V[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[0][2])*backup3V[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
								+ alpha*gaussKernel[0][2]*(frameV[(y+yld[m])*width+(x+xld[m])]);
						}
						// left
						if (  (x+xl[m])>=0 && 
							match[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i] == true  ){
								model3H[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[1][2])*backup3H[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
								+ alpha*gaussKernel[1][2]*(frameH[(y+yl[m])*width+(x+xl[m])]);
								model3S[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[1][2])*backup3S[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
								+ alpha*gaussKernel[1][2]*(frameS[(y+yl[m])*width+(x+xl[m])]);
								model3V[(y*3+j)*width*3+(x*3+i)] = 
									(1-alpha*gaussKernel[1][2])*backup3V[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
								+ alpha*gaussKernel[1][2]*(frameV[(y+yl[m])*width+(x+xl[m])]);
						} // if
					} // for
				} // for
			}  // if
		} // if
}

__global__ void initMean(float* modelH, float* modelS, float* modelV, 
	float* frameH, float* frameS, float* frameV,
	float* meanDistance,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				meanDistance[(y*3+j)*width*3+(x*3+i)] = 
					distance(pointFrame[0], pointFrame[1], pointFrame[2],
					modelH[(y*3+j)*width*3+(x*3+i)], 
					modelS[(y*3+j)*width*3+(x*3+i)],
					modelV[(y*3+j)*width*3+(x*3+i)]);
			}
		}
}

__global__ void meanSum(float* modelH, float* modelS, float* modelV, 
	float* frameH, float* frameS, float* frameV,
	float* meanDistance,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				meanDistance[(y*3+j)*width*3+(x*3+i)] = (
					meanDistance[(y*3+j)*width*3+(x*3+i)] +
					distance(pointFrame[0], pointFrame[1], pointFrame[2],
					modelH[(y*3+j)*width*3+(x*3+i)], 
					modelS[(y*3+j)*width*3+(x*3+i)],
					modelV[(y*3+j)*width*3+(x*3+i)])
					)/2;
			}
		}
}

__global__ void meanSumWithFilter(float* model1H, float* model1S, float* model1V,
	float* model2H, float* model2S, float* model2V,
	float* frameH, float* frameS, float* frameV,
	float* maxDistance,
	float* meanDistance,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = model1H[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = model1S[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = model1V[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int i2 = 0;
		float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist >= max)
				max = dist; 
		}

		if( max >= maxDistance[y*width+x] ){
			for (int j = 0; j < 3; ++j){
				for (int i = 0; i < 3; ++i){
					meanDistance[(y*3+j)*width*3+(x*3+i)] = (
						meanDistance[(y*3+j)*width*3+(x*3+i)] +
						distance(pointFrame[0], pointFrame[1], pointFrame[2],
						model2H[(y*3+j)*width*3+(x*3+i)], 
						model2S[(y*3+j)*width*3+(x*3+i)],
						model2V[(y*3+j)*width*3+(x*3+i)])
						)/2;
				}
			}
		} // if
}

__global__ void meanSumWithFilter2(float* model1H, float* model1S, float* model1V,
	float* model2H, float* model2S, float* model2V,
	float* model3H, float* model3S, float* model3V,
	float* frameH, float* frameS, float* frameV,
	float* maxDistance1,
	float* maxDistance2,
	float* meanDistance,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// used to calculate the distance
		float pointFrame[3];
		float pointModel[9][3];

		pointFrame[0] = frameH[y*width + x];
		pointFrame[1] = frameS[y*width + x];
		pointFrame[2] = frameV[y*width + x];

		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				pointModel[j*3+i][0] = model1H[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][1] = model1S[(y*3+j)*width*3+(x*3+i)];
				pointModel[j*3+i][2] = model1V[(y*3+j)*width*3+(x*3+i)];
			}
		}

		int i2 = 0;
		float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
			pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

		for (int i2 = 1; i2 < 3*3; ++i2){ 
			float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
			if (dist >= max)
				max = dist; 
		}

		if( max >= maxDistance1[y*width+x] ){
			for (int j = 0; j < 3; ++j){
				for (int i = 0; i < 3; ++i){
					pointModel[j*3+i][0] = model2H[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][1] = model2S[(y*3+j)*width*3+(x*3+i)];
					pointModel[j*3+i][2] = model2V[(y*3+j)*width*3+(x*3+i)];
				}
			}

			int i2 = 0;
			float max = distance(pointFrame[0], pointFrame[1], pointFrame[2],
				pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);

			for (int i2 = 1; i2 < 3*3; ++i2){ 
				float dist = distance(pointFrame[0], pointFrame[1], pointFrame[2],
					pointModel[i2][0], pointModel[i2][1], pointModel[i2][2]);
				if (dist >= max)
					max = dist; 
			}

			if( max>= maxDistance2[y*width+x] ){
				for (int j = 0; j < 3; ++j){
					for (int i = 0; i < 3; ++i){
						meanDistance[(y*3+j)*width*3+(x*3+i)] = (
							meanDistance[(y*3+j)*width*3+(x*3+i)] +
							distance(pointFrame[0], pointFrame[1], pointFrame[2],
							model3H[(y*3+j)*width*3+(x*3+i)], 
							model3S[(y*3+j)*width*3+(x*3+i)],
							model3V[(y*3+j)*width*3+(x*3+i)])
							)/2;
					}
				}
			} // if
		} // if
}

__global__ void detection(float* inputH, float* inputS, float* inputV,
	float* layer1H, float* layer1S, float* layer1V,
	float* layer2H, float* layer2S, float* layer2V,
	float* layer3H, float* layer3S, float* layer3V,
	float* thresholdLayer1, float* thresholdLayer2, float* thresholdLayer3,
	float* ouput,
	float* labelLayerMatch,
	bool* matchLayer1, bool* matchLayer2, bool* matchLayer3,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		float min1 = distance(inputH[y*width+x], inputS[y*width+x], inputV[y*width+x],
			layer1H[(y*3)*width*3+x*3], layer1S[(y*3)*width*3+x*3], layer1V[(y*3)*width*3+x*3]);
		int index1 = 0;
		for(int j = 0; j < 3; ++j){
			for(int i = 0; i < 3; ++i){
				float distTemp = distance(inputH[y*width+x], inputS[y*width+x], inputV[y*width+x],
					layer1H[(y*3+j)*width*3+(x*3+i)], 
					layer1S[(y*3+j)*width*3+(x*3+i)], layer1V[(y*3+j)*width*3+(x*3+i)]);
				if(distTemp <= min1){
					min1 = distTemp;
					index1 = j*3 + i;
				}
			}
		}

		for(int j = 0; j < 3; ++j){
			for(int i =0 ; i < 3; ++i){
				matchLayer1[(y*3+j)*width*3+(x*3+i)] = false;
				matchLayer2[(y*3+j)*width*3+(x*3+i)] = false;
				matchLayer3[(y*3+j)*width*3+(x*3+i)] = false;
			}
		}

		if( min1 <= thresholdLayer1[y*width+x] ){
			ouput[y*width+x] = 0;
			labelLayerMatch[y*width+x] = 1;
			matchLayer1[(y*3+mj[index1])*width*3+(x*3+mi[index1])] = true;
		}
		else{
			float min2 = distance(inputH[y*width+x], inputS[y*width+x], inputV[y*width+x],
				layer2H[(y*3)*width*3+x*3], layer2S[(y*3)*width*3+x*3], layer2V[(y*3)*width*3+x*3]);
			int index2 = 0;
			for(int j = 0; j < 3; ++j){
				for(int i = 0; i < 3; ++i){
					float distTemp = distance(inputH[y*width+x], inputS[y*width+x], inputV[y*width+x],
						layer2H[(y*3+j)*width*3+(x*3+i)], 
						layer2S[(y*3+j)*width*3+(x*3+i)], layer2V[(y*3+j)*width*3+(x*3+i)]);
					if(distTemp <= min2){
						min2 = distTemp;
						index2 = j*3 + i;
					}
				} // i
			} // j

			if( min2 <= thresholdLayer2[y*width+x] ){
				ouput[y*width+x] = 0;
				labelLayerMatch[y*width+x] = 2;
				matchLayer2[(y*3+mj[index2])*width*3+(x*3+mi[index2])] = true;
			}
			else{
				float min3 = distance(inputH[y*width+x], inputS[y*width+x], inputV[y*width+x],
					layer3H[(y*3)*width*3+x*3], layer3S[(y*3)*width*3+x*3], layer3V[(y*3)*width*3+x*3]);
				int index3 = 0;
				for(int j = 0; j < 3; ++j){
					for(int i = 0; i < 3; ++i){
						float distTemp = distance(inputH[y*width+x], inputS[y*width+x], inputV[y*width+x],
							layer3H[(y*3+j)*width*3+(x*3+i)], 
							layer3S[(y*3+j)*width*3+(x*3+i)], layer3V[(y*3+j)*width*3+(x*3+i)]);
						if(distTemp <= min3){
							min3 = distTemp;
							index3 = j*3 + i;
						}
					} // i
				} // j
				
				if( min3 <= thresholdLayer3[y*width+x] ){
					ouput[y*width+x] = 0;
					labelLayerMatch[y*width+x] = 3;
					matchLayer3[(y*3+mj[index2])*width*3+(x*3+mi[index2])] = true;
				}
				else{
					ouput[y*width+x] = 1;
					labelLayerMatch[y*width+x] = 0;
				} // else
			} // else
		} // else
}

__global__ void update(float* frameH, float* frameS, float* frameV,
	float* model1H, float* model1S, float* model1V,
	float* backup1H, float* backup1S, float* backup1V,
	float* model2H, float* model2S, float* model2V, 
	float* backup2H, float* backup2S, float* backup2V,
	float* model3H, float* model3S, float* model3V, 
	float* backup3H, float* backup3S, float* backup3V,
	float* labelLayerMatch,
	bool* matchLayer1, bool* matchLayer2, bool* matchLayer3,
	int width, int height, float alpha){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		// layer1
		if(labelLayerMatch[y*width+x] == 1){
			int m = 0;
			for(int j = 0; j < 3; ++j){
				for(int i = 0; i < 3; ++i){
					m = j*3+i;
					// center
					if(matchLayer1[(y*3+j)*width*3+(x*3+i)] == true){
						model1H[(y*3+j)*width*3+(x*3+i)] = 
							(1-alpha*gaussKernel[1][1])*backup1H[(y*3+j)*width*3+(x*3+i)]
						+ alpha*gaussKernel[1][1]*(frameH[y*width+x]);
						model1S[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup1S[(y*3+j)*width*3+(x*3+i)] 
						+ alpha*gaussKernel[1][1]*(frameS[y*width+x]);
						model1V[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup1V[(y*3+j)*width*3+(x*3+i)] + 
							alpha*gaussKernel[1][1]*(frameV[y*width+x]);
					}
					// left up
					if (  (x+xlu[m])>=0 && (y+ylu[m])>=0 && 
						matchLayer1[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] == true  ){
							model1H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup1H[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] 
							+ alpha*gaussKernel[2][2]*(frameH[(y+ylu[m])*width+(x+xlu[m])]);
							model1S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup1S[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameS[(y+ylu[m])*width+(x+xlu[m])]);
							model1V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup1V[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameV[(y+ylu[m])*width+(x+xlu[m])]);
					}
					// up
					if (  (y+yu[m])>=0 && 
						matchLayer1[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i] == true  ){
							model1H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup1H[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameH[(y+yu[m])*width+(x+xu[m])]);
							model1S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup1S[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameS[(y+yu[m])*width+(x+xu[m])]);
							model1V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup1V[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameV[(y+yu[m])*width+(x+xu[m])]);
					}
					// right up
					if (  (x+xru[m])<=width && (y+yru[m])>=0 && 
						matchLayer1[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] == true  ){
							model1H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup1H[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[2][0]*(frameH[(y+yru[m])*width+(x+xru[m])]);
							model1S[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[2][0])*backup1S[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] 
							+ alpha*gaussKernel[2][0]*(frameS[(y+yru[m])*width+(x+xru[m])]);
							model1V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup1V[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yru[m])*width+(x+xru[m])]);
					}
					// right
					if (  (x+xr[m])<=width && 
						matchLayer1[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i] == true  ){
							model1H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup1H[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameH[(y+yr[m])*width+(x+xr[m])]);
							model1S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup1S[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameS[(y+yr[m])*width+(x+xr[m])]);
							model1V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup1V[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameV[(y+yr[m])*width+(x+xr[m])]);
					}
					// right down
					if (  (x+xrd[m])<=width && (y+yrd[m])>=height && 
						matchLayer1[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i] ==true  ){
							model1H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup1H[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameH[(y+yrd[m])*width+(x+xrd[m])]);
							model1S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup1S[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameS[(y+yrd[m])*width+(x+xrd[m])]);
							model1V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup1V[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameV[(y+yrd[m])*width+(x+xrd[m])]);
					}
					// down
					if (  (y+yd[m])>=height && 
						matchLayer1[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i] == true  ){
							model1H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup1H[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameH[(y+yd[m])*width+(x+xd[m])]);
							model1S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup1S[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameS[(y+yd[m])*width+(x+xd[m])]);
							model1V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup1V[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameV[(y+yd[m])*width+(x+xd[m])]);
					}
					// left down7
					if (  (y+yld[m])>=height && (x+xld[m])>=0 && 
						matchLayer1[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i] == true  ){
							model1H[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[0][2])*backup1H[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameH[(y+yld[m])*width+(x+xld[m])]);
							model1S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup1S[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameS[(y+yld[m])*width+(x+xld[m])]);
							model1V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup1V[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yld[m])*width+(x+xld[m])]);
					}
					// left
					if (  (x+xl[m])>=0 && 
						matchLayer1[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i] == true  ){
							model1H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup1H[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameH[(y+yl[m])*width+(x+xl[m])]);
							model1S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup1S[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameS[(y+yl[m])*width+(x+xl[m])]);
							model1V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup1V[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameV[(y+yl[m])*width+(x+xl[m])]);
					}
				}
			}
		}

		// layer2
		if(labelLayerMatch[y*width+x] == 2){
			int m = 0;
			for(int j = 0; j < 3; ++j){
				for(int i = 0; i < 3; ++i){
					m = j*3+i;
					// center
					if(matchLayer2[(y*3+j)*width*3+(x*3+i)] == true){
						model2H[(y*3+j)*width*3+(x*3+i)] = 
							(1-alpha*gaussKernel[1][1])*backup2H[(y*3+j)*width*3+(x*3+i)]
						+ alpha*gaussKernel[1][1]*(frameH[y*width+x]);
						model2S[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup2S[(y*3+j)*width*3+(x*3+i)] 
						+ alpha*gaussKernel[1][1]*(frameS[y*width+x]);
						model2V[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup2V[(y*3+j)*width*3+(x*3+i)] + 
							alpha*gaussKernel[1][1]*(frameV[y*width+x]);
					}
					// left up
					if (  (x+xlu[m])>=0 && (y+ylu[m])>=0 && 
						matchLayer2[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2H[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] 
							+ alpha*gaussKernel[2][2]*(frameH[(y+ylu[m])*width+(x+xlu[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2S[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameS[(y+ylu[m])*width+(x+xlu[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup2V[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameV[(y+ylu[m])*width+(x+xlu[m])]);
					}
					// up
					if (  (y+yu[m])>=0 && 
						matchLayer2[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2H[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameH[(y+yu[m])*width+(x+xu[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2S[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameS[(y+yu[m])*width+(x+xu[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup2V[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameV[(y+yu[m])*width+(x+xu[m])]);
					}
					// right up
					if (  (x+xru[m])<=width && (y+yru[m])>=0 && 
						matchLayer2[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup2H[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[2][0]*(frameH[(y+yru[m])*width+(x+xru[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[2][0])*backup2S[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] 
							+ alpha*gaussKernel[2][0]*(frameS[(y+yru[m])*width+(x+xru[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup2V[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yru[m])*width+(x+xru[m])]);
					}
					// right
					if (  (x+xr[m])<=width && 
						matchLayer2[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2H[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameH[(y+yr[m])*width+(x+xr[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2S[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameS[(y+yr[m])*width+(x+xr[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup2V[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameV[(y+yr[m])*width+(x+xr[m])]);
					}
					// right down
					if (  (x+xrd[m])<=width && (y+yrd[m])>=height && 
						matchLayer2[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i] ==true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2H[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameH[(y+yrd[m])*width+(x+xrd[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2S[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameS[(y+yrd[m])*width+(x+xrd[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup2V[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameV[(y+yrd[m])*width+(x+xrd[m])]);
					}
					// down
					if (  (y+yd[m])>=height && 
						matchLayer2[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2H[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameH[(y+yd[m])*width+(x+xd[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2S[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameS[(y+yd[m])*width+(x+xd[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup2V[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameV[(y+yd[m])*width+(x+xd[m])]);
					}
					// left down7
					if (  (y+yld[m])>=height && (x+xld[m])>=0 && 
						matchLayer2[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[0][2])*backup2H[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameH[(y+yld[m])*width+(x+xld[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup2S[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameS[(y+yld[m])*width+(x+xld[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup2V[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yld[m])*width+(x+xld[m])]);
					}
					// left
					if (  (x+xl[m])>=0 && 
						matchLayer2[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i] == true  ){
							model2H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2H[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameH[(y+yl[m])*width+(x+xl[m])]);
							model2S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2S[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameS[(y+yl[m])*width+(x+xl[m])]);
							model2V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup2V[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameV[(y+yl[m])*width+(x+xl[m])]);
					}
				}
			}
		}

		// layer 3
		if(labelLayerMatch[y*width+x] == 3){
			int m = 0;
			for(int j = 0; j < 3; ++j){
				for(int i = 0; i < 3; ++i){
					m = j*3+i;
					// center
					if(matchLayer3[(y*3+j)*width*3+(x*3+i)] == true){
						model3H[(y*3+j)*width*3+(x*3+i)] = 
							(1-alpha*gaussKernel[1][1])*backup3H[(y*3+j)*width*3+(x*3+i)]
						+ alpha*gaussKernel[1][1]*(frameH[y*width+x]);
						model3S[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup3S[(y*3+j)*width*3+(x*3+i)] 
						+ alpha*gaussKernel[1][1]*(frameS[y*width+x]);
						model3V[(y*3+j)*width*3+(x*3+i)] =
							(1-alpha*gaussKernel[1][1])*backup3V[(y*3+j)*width*3+(x*3+i)] + 
							alpha*gaussKernel[1][1]*(frameV[y*width+x]);
					}
					// left up
					if (  (x+xlu[m])>=0 && (y+ylu[m])>=0 && 
						matchLayer3[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] == true  ){
							model3H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup3H[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i] 
							+ alpha*gaussKernel[2][2]*(frameH[(y+ylu[m])*width+(x+xlu[m])]);
							model3S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup3S[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameS[(y+ylu[m])*width+(x+xlu[m])]);
							model3V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][2])*backup3V[((y+ylu[m])*3+j)*width*3+(x+xlu[m])*3+i]
							+ alpha*gaussKernel[2][2]*(frameV[(y+ylu[m])*width+(x+xlu[m])]);
					}
					// up
					if (  (y+yu[m])>=0 && 
						matchLayer3[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i] == true  ){
							model3H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup3H[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameH[(y+yu[m])*width+(x+xu[m])]);
							model3S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup3S[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameS[(y+yu[m])*width+(x+xu[m])]);
							model3V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][1])*backup3V[((y+yu[m])*3+j)*width*3+(x+xu[m])*3+i]
							+ alpha*gaussKernel[2][1]*(frameV[(y+yu[m])*width+(x+xu[m])]);
					}
					// right up
					if (  (x+xru[m])<=width && (y+yru[m])>=0 && 
						matchLayer3[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] == true  ){
							model3H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup3H[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[2][0]*(frameH[(y+yru[m])*width+(x+xru[m])]);
							model3S[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[2][0])*backup3S[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i] 
							+ alpha*gaussKernel[2][0]*(frameS[(y+yru[m])*width+(x+xru[m])]);
							model3V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[2][0])*backup3V[((y+yru[m])*3+j)*width*3+(x+xru[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yru[m])*width+(x+xru[m])]);
					}
					// right
					if (  (x+xr[m])<=width && 
						matchLayer3[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i] == true  ){
							model3H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup3H[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameH[(y+yr[m])*width+(x+xr[m])]);
							model3S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup3S[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameS[(y+yr[m])*width+(x+xr[m])]);
							model3V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][0])*backup3V[((y+yr[m])*3+j)*width*3+(x+xr[m])*3+i]
							+ alpha*gaussKernel[1][0]*(frameV[(y+yr[m])*width+(x+xr[m])]);
					}
					// right down
					if (  (x+xrd[m])<=width && (y+yrd[m])>=height && 
						matchLayer3[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i] ==true  ){
							model3H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup3H[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameH[(y+yrd[m])*width+(x+xrd[m])]);
							model3S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup3S[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameS[(y+yrd[m])*width+(x+xrd[m])]);
							model3V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][0])*backup3V[((y+yrd[m])*3+j)*width*3+(x+xrd[m])*3+i]
							+ alpha*gaussKernel[0][0]*(frameV[(y+yrd[m])*width+(x+xrd[m])]);
					}
					// down
					if (  (y+yd[m])>=height && 
						matchLayer3[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i] == true  ){
							model3H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup3H[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameH[(y+yd[m])*width+(x+xd[m])]);
							model3S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup3S[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameS[(y+yd[m])*width+(x+xd[m])]);
							model3V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][1])*backup3V[((y+yd[m])*3+j)*width*3+(x+xd[m])*3+i]
							+ alpha*gaussKernel[0][1]*(frameV[(y+yd[m])*width+(x+xd[m])]);
					}
					// left down7
					if (  (y+yld[m])>=height && (x+xld[m])>=0 && 
						matchLayer3[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i] == true  ){
							model3H[(y*3+j)*width*3+(x*3+i)] =
								(1-alpha*gaussKernel[0][2])*backup3H[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameH[(y+yld[m])*width+(x+xld[m])]);
							model3S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup3S[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameS[(y+yld[m])*width+(x+xld[m])]);
							model3V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[0][2])*backup3V[((y+yld[m])*3+j)*width*3+(x+xld[m])*3+i]
							+ alpha*gaussKernel[0][2]*(frameV[(y+yld[m])*width+(x+xld[m])]);
					}
					// left
					if (  (x+xl[m])>=0 && 
						matchLayer3[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i] == true  ){
							model3H[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup3H[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameH[(y+yl[m])*width+(x+xl[m])]);
							model3S[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup3S[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameS[(y+yl[m])*width+(x+xl[m])]);
							model3V[(y*3+j)*width*3+(x*3+i)] = 
								(1-alpha*gaussKernel[1][2])*backup3V[((y+yl[m])*3+j)*width*3+(x+xl[m])*3+i]
							+ alpha*gaussKernel[1][2]*(frameV[(y+yl[m])*width+(x+xl[m])]);
					}
				}
			}
		}
}
	
// with training
__global__ void calculateThreshold(float* meanValue, float* maxValue,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		float tempMax = meanValue[(y*3)*width*3+(x*3)];
		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				if( meanValue[(y*3+j)*width*3+(x*3+i)]>=tempMax )
					tempMax = meanValue[(y*3+j)*width*3+(x*3+i)];
			}
		}
		maxValue[y*width+x] = tempMax;
}

// without training, we set the minimum value of tao as 0.06
__global__ void calculateThresholdWithoutTraining(float* meanValue, float* maxValue,
	int width){
		int x = blockDim.x*blockIdx.x + threadIdx.x;
		int y = blockDim.y*blockIdx.y + threadIdx.y;

		float tempMax = meanValue[(y*3)*width*3+(x*3)];
		for (int j = 0; j < 3; ++j){
			for (int i = 0; i < 3; ++i){
				if( meanValue[(y*3+j)*width*3+(x*3+i)]>=tempMax )
					tempMax = meanValue[(y*3+j)*width*3+(x*3+i)];
			}
		}
		maxValue[y*width+x] = tempMax;

		// can be adjusted by hand
		if( maxValue[y*width+x]<= 0.06 )
			maxValue[y*width+x] = 0.06;
}

void help(){
	cout<<">----------------------------------------------------------------------------------------------------------------<"<<endl<<endl;
	cout<<"The command format is:"<<endl<<endl;
	cout<<"1. smsom train <start_frame_number> <end_frame_number> <input_file_name> <output_file_name>"<<endl<<endl;
	cout<<"2. smsom train <start_frame_number> <end_frame_number> <input_file_name>"<<endl<<endl;
	cout<<"3. smsom nottrain <input_file_name> <output_file_name>"<<endl<<endl;
	cout<<"4. smsom nottrain <input_file_name> <output_file_name>"<<endl<<endl;
	cout<<"Please see http://zhaozj89.github.io/SMSOM/ for more details"<<endl<<endl;
	cout<<"Press 'q' to exit"<<endl<<endl;
	cout<<">----------------------------------------------------------------------------------------------------------------<"<<endl;
}

// 
float c1 = 1;
float c2 = 0.03;
float alphaLearning = c1*4; // c1/max weight of the Gaussian kernel
float alphaAdaption = c2*4; // c2/max weight of the Gaussian kernel
int startFrame, endFrame;
int initFrame = 1;

//
bool IsTraining;
bool IsOuput;
char fileName[200];
char outputFileName[200];
char path[200];
char outputPath[200];

int main(int argv, char* argc[]){

	//
	if(argv < 3){
		help();
		return 0;
	}

	//
	string p2(argc[1]);
	string tempTrain = "train";
	string tempNottrain = "nottrain";
	if(p2 == tempTrain) IsTraining = true;
	else if(p2 == tempNottrain) IsTraining = false;
	else{
		help();
		return 0;
	}

	//
	if(IsTraining == true){
		if(argv == 6){
			startFrame = atoi(argc[2]);
			endFrame = atoi(argc[3]);
			strcpy(path, argc[4]);
			strcpy(outputPath, argc[5]);
			IsOuput = true;
		}
		else if(argv == 5){
			startFrame = atoi(argc[2]);
			endFrame = atoi(argc[3]);
			strcpy(path, argc[4]);
			IsOuput = false;
		}
		else{
			help();
			return 0;
		}
	}
	else{
		if(argv == 4){
			strcpy(path, argc[2]);
			strcpy(outputPath, argc[3]);
			IsOuput = true;
		}
		else if(argv == 3){
			strcpy(path, argc[2]);
			IsOuput = false;
		}
		else{
			help();
			return 0;
		}
	}

	// test whether input is legal
	{
		if(startFrame > endFrame){
			cout<<"<start_frame_number> or <end_frame_number> is  illegal, please retry!"<<endl;
			return 0;
		}
		Mat frame;
		sprintf(fileName, path, initFrame); // read the first frame
		frame = imread(fileName, CV_LOAD_IMAGE_COLOR);
		if(frame.empty()){
			cout<<"<input_file_name> is illegal, please retry!"<<endl;
			return 0;
		}
	}

	Mat frame;
	sprintf(fileName, path, initFrame); // read the first frame
	frame = imread(fileName, CV_LOAD_IMAGE_COLOR);
	int width = frame.cols;
	int height = frame.rows;

	Mat frameFloat;
	Mat frameFloat2;
	frameFloat.create(height, width, CV_32FC3);
	frameFloat2.create(height, width, CV_32FC3);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);

	vector<Mat> input(3);
	input[0].create(height, width, CV_32FC1);
	input[1].create(height, width, CV_32FC1);
	input[2].create(height, width, CV_32FC1);
	split(frameFloat2, input);

	vector<float*> gpuInput(3);
	vector<float*> gpuLayer1(3);
	vector<float*> gpuLayer1Backup(3);
	bool* gpuMatch1;
	float* gpuOutput;
	float* gpuOutputBackup;

	Mat output;
	Mat outputFile;
	output.create(height, width, CV_32FC1);
	outputFile.create(height, width, CV_8UC3);

	for(int i = 0; i < 3; ++i){
		cudaMalloc((void**)&gpuInput[i], width*height*sizeof(float));
		cudaMalloc((void**)&gpuLayer1[i], width*height*3*3*sizeof(float));
		cudaMalloc((void**)&gpuLayer1Backup[i], width*height*3*3*sizeof(float));
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	}
	cudaMalloc((void**)&gpuMatch1, width*height*3*3*sizeof(bool));
	cudaMalloc((void**)&gpuOutput, width*height*sizeof(float));
	cudaMalloc((void**)&gpuOutputBackup, width*height*sizeof(float));

	dim3 grid( (width-1)/16+1, (height-1)/16+1, 1 );
	dim3 block(16, 16, 1);

	// Stacked Multi-layer Self Organizing Map Background Model (in this code, 3 layers)
	// A layer is composed of 2 parts: train and log

	// initialize layer 1
	for(int i = 0; i < 3; ++i)
		initLayer<<<grid, block>>>(gpuInput[i], gpuLayer1[i], width);

	// train layer 1
	cout<<"start training layer 1 ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(gpuLayer1Backup[j], gpuLayer1[j], width*height*3*3*sizeof(float), cudaMemcpyDeviceToDevice);
		}

		compete<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMatch1, width);
		cooperate<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer1Backup[0], gpuLayer1Backup[1], gpuLayer1Backup[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMatch1,
			width, height, alphaLearning);
	}

	// log layer 1
	float* gpuMeanDistance1;
	float* gpuMaxDistance1;
	cudaMalloc((void**)&gpuMeanDistance1, width*height*3*3*sizeof(float));
	cudaMalloc((void**)&gpuMaxDistance1, width*height*sizeof(float));
	
	// first frame
	sprintf(fileName, path, initFrame);
	frame = imread(fileName);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
	split(frameFloat2, input);
	for(int i = 0; i < 3; ++i)
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	initMean<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
		gpuInput[0], gpuInput[1], gpuInput[2],
		gpuMeanDistance1, width);

	cout<<"calculate the thresholds for detection and layer 2 input ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
		}

		meanSum<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMeanDistance1, width);
	}

	//
	if(IsTraining == true){
		calculateThreshold<<<grid, block>>>(gpuMeanDistance1, 
			gpuMaxDistance1, width);
	}
	else{
		calculateThresholdWithoutTraining<<<grid, block>>>(gpuMeanDistance1, 
			gpuMaxDistance1, width);
	}


	// train layer 2
	vector<float*> gpuLayer2(3);
	vector<float*> gpuLayer2Backup(3);
	bool* gpuMatch2;
	for (int i = 0; i < 3; ++i){
		cudaMalloc((void**)&gpuLayer2[i], width*height*3*3*sizeof(float));
		cudaMalloc((void**)&gpuLayer2Backup[i], width*height*3*3*sizeof(float));
	}
	cudaMalloc((void**)&gpuMatch2, width*height*3*3*sizeof(bool));

	// first frame
	sprintf(fileName, path, initFrame);
	frame = imread(fileName);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
	split(frameFloat2, input);
	for(int i = 0; i < 3; ++i)
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	for(int i = 0; i < 3; ++i)
		initLayer<<<grid, block>>>(gpuInput[i], gpuLayer2[i], width); // TODO: better initialization

	cout<<"start training layer 2 ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(gpuLayer2Backup[j], gpuLayer2[j], width*height*3*3*sizeof(float), cudaMemcpyDeviceToDevice);
		}

		competeWithFilter<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMaxDistance1,
			gpuMatch2, width);
		cooperateWithFilter<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuLayer2Backup[0], gpuLayer2Backup[1], gpuLayer2Backup[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMaxDistance1,
			gpuMatch2,
			width, height, alphaLearning);
	}

	// log layer 2
	float* gpuMeanDistance2;
	float* gpuMaxDistance2;
	cudaMalloc((void**)&gpuMeanDistance2, width*height*3*3*sizeof(float));
	cudaMalloc((void**)&gpuMaxDistance2, width*height*sizeof(float));

	// first frame
	sprintf(fileName, path, initFrame);
	frame = imread(fileName);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
	split(frameFloat2, input);
	for(int i = 0; i < 3; ++i)
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	initMean<<<grid, block>>>(gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
		gpuInput[0], gpuInput[1], gpuInput[2],
		gpuMeanDistance2, width);

	cout<<"calculate the thresholds for detection and layer 3 input ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
		}

		meanSumWithFilter<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMaxDistance1,
			gpuMeanDistance2, width);
	}

	//
	if(IsTraining == true){
		calculateThreshold<<<grid, block>>>(gpuMeanDistance1, 
			gpuMaxDistance1, width);
	}
	else{
		calculateThresholdWithoutTraining<<<grid, block>>>(gpuMeanDistance1, 
			gpuMaxDistance1, width);
	}

	// train layer 3
	vector<float*> gpuLayer3(3);
	vector<float*> gpuLayer3Backup(3);
	bool* gpuMatch3;
	for (int i = 0; i < 3; ++i){
		cudaMalloc((void**)&gpuLayer3[i], width*height*3*3*sizeof(float));
		cudaMalloc((void**)&gpuLayer3Backup[i], width*height*3*3*sizeof(float));
	}
	cudaMalloc((void**)&gpuMatch3, width*height*3*3*sizeof(bool));

	// first frame
	sprintf(fileName, path, initFrame);
	frame = imread(fileName);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
	split(frameFloat2, input);
	for(int i = 0; i < 3; ++i)
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	for(int i = 0; i < 3; ++i)
		initLayer<<<grid, block>>>(gpuInput[i], gpuLayer3[i], width); // TODO: better initialization

	cout<<"start training layer 3 ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(gpuLayer3Backup[j], gpuLayer3[j], width*height*3*3*sizeof(float), cudaMemcpyDeviceToDevice);
		}

		competeWithFilter2<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuLayer3[0], gpuLayer3[1], gpuLayer3[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMaxDistance1,
			gpuMaxDistance2,
			gpuMatch3, width);
		cooperateWithFilter2<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuLayer3[0], gpuLayer3[1], gpuLayer3[2],
			gpuLayer3Backup[0], gpuLayer3Backup[1], gpuLayer3Backup[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMaxDistance1,
			gpuMaxDistance2,
			gpuMatch3,
			width, height, alphaLearning);
	}

	// log layer 3
	float* gpuMeanDistance3;
	float* gpuMaxDistance3;
	cudaMalloc((void**)&gpuMeanDistance3, width*height*3*3*sizeof(float));
	cudaMalloc((void**)&gpuMaxDistance3, width*height*sizeof(float));

	// first frame
	sprintf(fileName, path, initFrame);
	frame = imread(fileName);
	frame.convertTo(frameFloat, CV_32FC3);
	frameFloat *= 1./255;
	cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
	split(frameFloat2, input);
	for(int i = 0; i < 3; ++i)
		cudaMemcpy(gpuInput[i], input[i].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
	initMean<<<grid, block>>>(gpuLayer3[0], gpuLayer3[1], gpuLayer3[2],
		gpuInput[0], gpuInput[1], gpuInput[2],
		gpuMeanDistance3, width);

	cout<<"calculate the thresholds for detection and layer 4 input ... ..."<<endl;
	for(int i = startFrame; i <= endFrame; ++i){
		if(i%100 == 0)
			cout<<"processing the "<<i<<"th image ... ..."<<endl;
		sprintf(fileName, path, i);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
		}

		meanSumWithFilter2<<<grid, block>>>(gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuLayer3[0], gpuLayer3[1], gpuLayer3[2],
			gpuInput[0], gpuInput[1], gpuInput[2],
			gpuMaxDistance1,
			gpuMaxDistance2,
			gpuMeanDistance3, width);
	}

	//
	if(IsTraining == true){
		calculateThreshold<<<grid, block>>>(gpuMeanDistance1, 
			gpuMaxDistance1, width);
	}
	else{
		calculateThresholdWithoutTraining<<<grid, block>>>(gpuMeanDistance1, 
			gpuMaxDistance1, width);
	}

	//////////////////////////////////////////////////////////////////////////
	////DEBUG
	//vector<Mat> tempLayer(3);
	//char tempName[200];
	//for(int i = 0; i < 3; ++i){
	//	tempLayer[i].create(height*3, width*3, CV_32FC1);
	//	cudaMemcpy(tempLayer[i].data, gpuLayer1[i], width*height*3*3*sizeof(float), cudaMemcpyDeviceToHost);
	//	namedWindow("temp", 1);
	//	imshow("temp", tempLayer[i]);
	//	waitKey(0);
	//	sprintf(tempName, "layer1_%d.png", i);
	//	imwrite(tempName, tempLayer[i]);
	//}

	//for(int i = 0; i < 3; ++i){
	//	tempLayer[i].create(height*3, width*3, CV_32FC1);
	//	cudaMemcpy(tempLayer[i].data, gpuLayer2[i], width*height*3*3*sizeof(float), cudaMemcpyDeviceToHost);
	//	namedWindow("temp", 1);
	//	imshow("temp", tempLayer[i]);
	//	waitKey(0);
	//	sprintf(tempName, "layer2_%d.png", i);
	//	imwrite(tempName, tempLayer[i]);
	//}
	//for(int i = 0; i < 3; ++i){
	//	tempLayer[i].create(height*3, width*3, CV_32FC1);
	//	cudaMemcpy(tempLayer[i].data, gpuLayer3[i], width*height*3*3*sizeof(float), cudaMemcpyDeviceToHost);
	//	namedWindow("temp", 1);
	//	imshow("temp", tempLayer[i]);
	//	waitKey(0);
	//	sprintf(tempName, "layer3_%d.png", i);
	//	imwrite(tempName, tempLayer[i]);
	//}

	//vector<Mat> tempMax(3);
	//for(int i = 0; i < 3; ++i){
	//	tempMax[i].create(height, width, CV_32FC1);
	//}
	//cudaMemcpy(tempMax[0].data, gpuMaxDistance1, height*width*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(tempMax[1].data, gpuMaxDistance2, height*width*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(tempMax[2].data, gpuMaxDistance3, height*width*sizeof(float), cudaMemcpyDeviceToHost);
	//char tempName[200];
	//vector<Mat> tempMaxInt(3);
	//for (int i = 0; i < 3; ++i){
	//	namedWindow("temp", 1);
	//	imshow("temp", tempMax[i]);
	//	waitKey(0);
	//	//tempMaxInt[i].create(height, width, CV_8UC1);
	//	//tempMax[i] *= 255;
	//	//tempMax[i].convertTo(tempMaxInt[i], CV_8UC1);
	//	//sprintf(tempName, "max_%d.png", i);
	//	//imwrite(tempName, tempMaxInt[i]);
	//}

	//for(int k = 0; k < 3; ++k){
	//	float* p;
	//	float maxV = -10.0;
	//	for(int i = 0; i < height; ++i){
	//		p = tempMax[k].ptr<float>(i);
	//		for(int j = 0; j < width; ++j){
	//			if( p[j] >= maxV )
	//				maxV = p[j];
	//		}
	//	}

	//	cout<<maxV<<endl;
	//}

	//return 0;
	//////////////////////////////////////////////////////////////////////////


	// start detection and update SMSOM on-line
	float* gpuLabelLayerMatch;
	cudaMalloc((void**)&gpuLabelLayerMatch, width*height*sizeof(float));
	cout<<"start detecting the foreground on-line ... ..."<<endl;
	char key = NULL;
	int frameNum = endFrame + 1;
	// clock_t startTime = clock();
	namedWindow("foreground", 1);
	// for(frameNum = 165; frameNum <= 300; ++frameNum){
	while (key != 'q'){
		++frameNum;
		if(frameNum%100 == 0)
			cout<<"processing the "<<frameNum<<"th image ... ..."<<endl;
		sprintf(fileName, path, frameNum);
		frame = imread(fileName);
		if(frame.empty()){
			cout<<"There are no images"<<endl;
			return 0;
		}
		frame.convertTo(frameFloat, CV_32FC3);
		frameFloat *= 1./255;
		cvtColor(frameFloat, frameFloat2, CV_BGR2HSV);
		split(frameFloat2, input);

		for(int j = 0; j < 3; ++j){
			cudaMemcpy(gpuInput[j], input[j].data, width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(gpuLayer1Backup[j], gpuLayer1[j], width*height*3*3*sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gpuLayer2Backup[j], gpuLayer2[j], width*height*3*3*sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(gpuLayer3Backup[j], gpuLayer3[j], width*height*3*3*sizeof(float), cudaMemcpyDeviceToDevice);
		}

		detection<<<grid, block>>>(gpuInput[0], gpuInput[1], gpuInput[2],
			gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuLayer3[0], gpuLayer3[1], gpuLayer3[2],
			gpuMaxDistance1, gpuMaxDistance2, gpuMaxDistance3,
			gpuOutput,
			gpuLabelLayerMatch,
			gpuMatch1, gpuMatch2, gpuMatch3,
			width);
		update<<<grid, block>>>(gpuInput[0], gpuInput[1], gpuInput[2],
			gpuLayer1[0], gpuLayer1[1], gpuLayer1[2],
			gpuLayer1Backup[0], gpuLayer1Backup[1], gpuLayer1Backup[2],
			gpuLayer2[0], gpuLayer2[1], gpuLayer2[2],
			gpuLayer2Backup[0], gpuLayer2Backup[1], gpuLayer2Backup[2],
			gpuLayer3[0], gpuLayer3[1], gpuLayer3[2],
			gpuLayer3Backup[0], gpuLayer3Backup[1], gpuLayer3Backup[2],
			gpuLabelLayerMatch,
			gpuMatch1, gpuMatch2, gpuMatch3,
			width, height, alphaAdaption
			);

		//Mat tempLabel;
		//tempLabel.create(height, width, CV_32FC1);
		//cudaMemcpy(tempLabel.data, gpuLabelLayerMatch, width*height*sizeof(float), cudaMemcpyDeviceToHost);
		//tempLabel /= 3.0;
		//namedWindow("temp", 1);
		//imshow("temp", tempLabel);
		//waitKey(0);

		cudaMemcpy(output.data, gpuOutput, width*height*sizeof(float), cudaMemcpyDeviceToHost);
		imshow("foreground", output);
		if(IsOuput == true){
			sprintf(outputFileName, outputPath, frameNum);
			output *= 255;
			output.convertTo(outputFile, CV_8UC3);
			imwrite(outputFileName, outputFile);
		}
		key = waitKey(1);
	}
	//clock_t endTime = clock();
	//cout<<"startTime="<<startTime<<endl;
	//cout<<"endTime="<<endTime<<endl;
	//cout<<"speed="<<(double)(endTime-startTime)/CLOCKS_PER_SEC<<endl;
	//DEBUG
	//for(int i = 0; i < 3; ++i){
	//	Mat outputTemp;
	//	outputTemp.create(height*3, width*3, CV_32FC1);
	//	cudaMemcpy(outputTemp.data, gpuLayer2[i], width*height*3*3*sizeof(float), cudaMemcpyDeviceToHost);
	//	namedWindow("layer2", 1);
	//	imshow("layer2", outputTemp/360);
	//	waitKey(0);
	//}


	//DEBUG
	//cudaMemcpy(output.data, gpuMaxDistance1, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//namedWindow("max", 1);
	//imshow("max", output);
	//waitKey(0);

	//cudaMemcpy(output.data, gpuThreshold1, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//namedWindow("threshold", 1);
	//imshow("threshold", output);
	//waitKey(0);
	return 0;
}