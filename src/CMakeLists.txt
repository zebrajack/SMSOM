project( smsom )

cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30, code=sm_30)

file(GLOB cu *.cu)

cuda_add_executable( smsom ${cu} )
target_link_libraries( smsom ${OpenCV_LIBS} )