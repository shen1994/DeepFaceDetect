# DeepFaceDetection

## 0. 效果展示  
![image](https://github.com/shen1994/DeepFaceDetect/raw/master/show/DeepFaceDetect.jpg)

## 1. 数据集及工具  
> * [人脸数据集 Wider Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

## 2. 运行命令  
> * 2.0 执行`python ssd_maker.py`制作数据集索引  
> * 2.1 执行`python train.py`训练数据  
> * 2.2 执行`Python convert.py`转换模型  
> * 2.3 执行`python test.py`测试单张图片数据  

## 3. 参考链接  
> * [SSD 论文](https://arxiv.org/abs/1512.02325)
> * [keras face 代码参考](https://github.com/bruceyang2012/Face-detection-with-mobilenet-ssd)
> * [keras ssd 代码参考](https://github.com/rykov8/ssd_keras)
> * [SSD 论文详解](https://blog.csdn.net/a8039974/article/details/77592395)   

## 4. 问题   
> * 对图片大尺度旋转识别不精准  
> * 图片锁定框与图片本身边框对其，这是不对的，比如人脸旋转一定角度，画出的边框也应该是斜的（参考face++人脸检测效果）  

## 5. 更新  
> * 修复test.py对单张图片的处理方式
