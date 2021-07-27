pycharm运行项目readme:
    先运行pre_process.py
    注意需要自己手动创建好train-clean-100-npy这个文件夹.否则不跑.
    
    docker里面无法用多进程
    所以用pre_process3.py替代即可.
    然后train.py即可.








jupyter 服务器docker部署 readme:

    docker pull zhangbo2008/jiaofu3
    docker run -ti --user root  -p 8888:8888 zhangbo2008/jiaofu3   /bin/bash
    然后jupyter notebook --allow-root
    
    之后任意电脑输入 上面服务器ip:8888即可打开
    然后密码是1.
    打开目录是服务器的/opt/app-root/src/
    所以把自己项目docker cp进这个目录即可.

    修改默认目录:
    找到并打开，找到jupyter_notebook_config.py文件中 c.NotebookApp.notebook_dir = u'D:\\ai_job'改成自己的项目路径，





##深度学习声纹识别系统


数据集: [LibriSpeech](http://www.openslr.org/12/)  
参考论文: "Deep Speaker: an End-to-End Neural Speaker Embedding System" https://arxiv.org/pdf/1705.02304.pdf  
该代码使用librispeech train clean数据集进行训练，使用librispeech test clean数据集进行测试。在我的代码librispeech数据集约1%的错误率。

##关于代码

train.py

这是主文件。该文件对模型进行训练，然后保存模型，并对每个具体步骤的结果进行评估。

models.py

这是本项目中使用的模型的实现。它包括三个模型，CNN模型（类似于本文的CNN），GRU模型（类似于本文的GRU），第三个模型是简化的simple-CNN模型。

select_batch.py

网络的困难batch样本生成。这是本实验的核心之一。

triplet_loss.py

这是用于计算网络训练的triplet_loss的代码。

test_model.py

这是一个评估（测试）模型的代码。

eval_matrics.py

该文件包含相等的错误率、f-度量、准确度和评估部分使用的其他度量。

pretaining.py

这是softmax分类的预训练代码。

pre_process.py

该代码实现了对语音数据的读取、静音滤波、提取fbank特征，并将提取的特征保存为.npy格式。

##结果

该代码使用librispeech train clean数据集进行训练，使用librispeech test clean数据集进行测试。在我的代码中，librispeech数据集使用CNN约1%的错误率。
  
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/raw/master/demo/loss.png"  width="400" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/raw/master/demo/EER.png" width="400" ></div>  
#实验流程运行



pip install tensorflow==1.5

先运行pre_process.py

train.py      模型会进行训练和评测

test_model.py 进行测试.

