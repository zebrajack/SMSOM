# How to use


### For Windows users:

If you have a working environment of OpenCV 2.4.5, CUDA 5.0, Visual Studio 2010, Windows 7 (64 bit), then you can use this repository directly. The executable is **.\Debug\smsom.exe**; therefore, you should first use **cmd** in Windows to navigate to the directory **Debug**. Please ENSURE smsom can find OpenCV library, and you have CUDA compatible GPU installed in your computer.


Then you have two options:

* If you have foreground free traning images, then execute:

  `smsom train <start_frame_number> <end_frame_number> <input_file_name> <output_file_name>`
    
  where `<start_frame_number>` and `<end_frame_number>` stand for the index range of the training images; `<input_file_name>`     is the format of the input image's name, and the last parameter `<output_file_name>` is optional, if you omit it, then the output images are just shown in your screen, but not stored in your computer. 

  For example, if I put the input images in: **E:\Data\input\**, 
  the image files' name format is: **in000001.jpg** (any number), and I use 1-100 images to train the model, then I can execute:

  `smsom train 1 100 E:\\Data\\input\\in%06d.jpg E:\\Data\\results\\bin%06d.jpg`

  where I put the result images in **E:\Data\results**.

  or

  `smsom train 1 100 E:\\Data\\input\\in%06d.jpg`

  where I do not store the output images.
  
* If you do not have foreground free training images, you can execute:
  
  `smsom nottrain <input_file_name> <output_file_name>`

  where the meanings of `<input_file_name>` and `<output_file_name>` (optional) are the same as the previous case. In this situation, we set the threshold tau=0.06 (see **[1]** for more details). 
 
### For Linux users:


You have to build yourself. Please ENSURE you have installed OpenCV and CUDA. See **[1]** for how to use CUDA on Linux platform. Then you can navigate to **./src/**, and execute the following commands in order:

`cmake .`

`make`

The usage of the generated executable `smsom` is the same as the commands shown previously in Windows. (You may use `./smsom <parameters>` instead of `smsom <parameters>`)

------------------------------------------------------------

# Demos

Some demo scripts are shown as follows (assuming you have decompressed dataset **[2]** in **E:\Data\**):

* fountain01: ``smsom train 1 399 E:\\Data\\dynamicBackground\\fountain01\\input\\in%06d.jpg``
* highway: ``smsom nottrain E:\\Data\\baseline\\highway\\input\\in%06d.jpg``
* traffic: ``smsom train 129 200 E:\\Data\\cameraJitter\\traffic\\input\\in%06d.jpg``
* ladeSide: ``smsom train 1 999 E:\\Data\\thermal\\lakeSide\\input\\in%06d.jpg``

------------------------------------------------------------

# Post-processing and quantitative evaluation

You should use the Matlab script **.\tools\median_filter.m** to do a 5X5 median filtering as suggested in **[2]**.

------------------------------------------------------------

# References

[1] http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3DimlP7Yp

[2] http://www.changedetection.net/
