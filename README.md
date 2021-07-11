## Steganalysis-StegoRemoval

​		该项目为本人的本科毕业设计，主要任务为实现**图像隐写分析以及隐写去除，其中隐写分析采用SRNet网络模型，隐写去除采用DDSP网络模型**。

​		项目中有4个文件夹，分别为: `0.SRNet`、`1.GUI`、`2.DDSP`、`3.SRNet` 其中`0.SRNet`为图像隐写分析，使用Jessica教授的官方源码，框架为tensorflow；`1.GUI`为隐写嵌入以及隐写分析可视化演示系统，由PyQ5实现；`2.DDSP`为图像隐写去除，pytorch实现；`3.SRNet`为图像隐写分析，pytorch实现。其中自己复现的SRNet网络模型其性能弱于官方代码。

​	该项目总代码在4600左右，最终虽然没拿到优秀本科生毕业论文 (Wu ~~~)，但是也拿到了95分。

### 隐写分析	

​		本项目隐写分析中使用的隐写术为:  S-UNIWARD、HUGO、WOW三种图像空域隐写算法，采用的隐写嵌入率为：0.4bpp、0.7bpp和1.0bpp三种。采用的隐写分析模型是2018年Jessica教授团队提出的SRNet隐写分析网络模型，关于网络模型此处不赘述，这里直接粘贴知乎的一篇帖子: https://zhuanlan.zhihu.com/p/362127299，SRNet隐写分析网络模型论文地址: https://ieeexplore.ieee.org/document/8470101. 隐写分析使用的BOSSBase数据集和隐写术的下载地址为: http://dde.binghamton.edu/download/stego_algorithms/，该页面可以下载BOSSBase1.01版数据集以及空域和JPEG域两大类隐写算法。隐写分析官方代码下载页面: http://dde.binghamton.edu/download/feature_extractors/。
​		隐写分析中为了进一步提升图像隐写分析的性能，本项目还将CBAM注意力机制和原始SRNet网络模型相结合，实验结果表明将CBAM注意力机制添加到SRNet网络模型中后，网络在某些嵌入率和隐写术中有性能的提高，但是对有些嵌入率和隐写术其性能还不如原始SRNet网络的性能。CBAM注意力机制论文地址为: https://arxiv.org/abs/1807.06521.

### 隐写去除

​		隐写去除采用的是DDSP模型，DDSP模型本质上是一个GAN网络，和SRGAN网络的结构非常类似，只不过DDSP网络的Generator是一个自编码器Autoencoder，在训练模型的过程中需要先训练自编码器，当自编码器收敛之后，再代入到GAN网络框架中进行对抗训练，GAN网络的鉴别器Discriminator是一个普通的卷积神经网络，主要是判别输入的图片是真实的图片还是自编器生成的图片，用于提高Autoencoder生成图片的视觉质量。DDSP隐写去除，个人认为更准确的描述是隐写破坏，也就破化之前嵌入的信息。

​		DDSP隐写去除模型作者没有公开实现代码，故本项目中隐写去除代码为小编本人独立实现（虽然效果比不上论文中描述的效果）DDSP论文地址为: https://arxiv.org/abs/1912.10070

### 演示系统

​		为了更好的演示如何实现隐写嵌入和隐写分析，使用PyQt5编写了可视化界面，调用现有的隐写术和训练好的隐写分析模型进行操作。

​		





