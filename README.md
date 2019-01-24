# Real time age recognition - Back End

*Authors*: Damian Gutowski, Piotr Wawrzyniak, Daria Hubernatorova

This project was created during Case Studies 2019 Winter semester https://github.com/pbiecek/CaseStudies2019W.

## 1. Technologies used
 - Python
 - [numpy](http://www.numpy.org/)
 - [Flask](http://flask.pocoo.org/)
 - [Pillow](https://github.com/python-pillow/Pillow/)
 - [Face Recongnition](https://github.com/ageitgey/face_recognition)
 - [Keras](https://keras.io/)
  - [TensorFlow](https://www.tensorflow.org/)
 
 ## 2. Instalation
 
1. Install Install CMake for Windows
2. Install TensorFlow.  
If your graphic card support CUDA platform, you can install TensorFlow Gpu.  
But you have to also install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)  
You can see what GPU cards is supports CUDA platform on [wiki](https://en.wikipedia.org/wiki/CUDA) 
4. Install adidtional required libraries:

```
pip install Pillow==4.0.0 # For python 3.6
pip install dlib
pip install face_recognition
pip install Flask

#For development you can also install Flask cors
pip insatll flask-cors
 ```
 
## 3. Self-hosting
To start the server, execute following command.

```
flask run
```
