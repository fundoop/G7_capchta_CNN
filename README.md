# G7_capchta_CNN
利用卷积神经网络（CNN）解决图片验证码登录问题。从验证码下载、标注、图片简单处理、神经网络构建到自动化批量标注、人工校验、模型迭代，准确率96%。最后模型应用于自动登录网站爬取数据

# 工具

Python

# 步骤

1、获取验证码。登录标的网站A，获取验证码路径，通常为固定路径+随机数字，本例中有Cookie验证，获取Cookie，通过cookie及随机生成数字（可以不随机，防止服务端特殊验证机制）获取验证码，第一批获取200张（因为要人工标注，费时费力）。

2、标注图片。对图片进行人工识别验证码，并进行标注，如果有多个图片对应同一个验证码，在后面加上额外字符串如“（1）”“（2）”等，因后期获取真实验证码只取到文件名的前五个字符

3、模型构建。本例使用两层卷积层，一层池化层，重复3次；Flatten层（数据一维化）；Dropout层（随机断开神经网络，降低过拟合，“防止过拟合”感觉并不合适，过拟合无法防止，只能降低其影响，当然早期代码或注释理解不透彻望理解！！）；全连接层，本例验证码为五位，因此取五个分类器，每个分类器取验证码可能包含的字符个数，本例为29个字符，因此将得到5*29 个数值，每个数值分别代表该位置取对应字符的概率。

4、模型的运行和调优。经过上述过程，模型基本搭建完成，现运行。取优化器Adadelta，学习率0.1，第一次运行效果很低，单字符识别率不到5%，验证码识别率为0，意料中事！经过观察得到，验证码中存在干扰线，故使用filter滤波器进行去噪声，经过调试，取3*3 filter滤波器，阈值为6，函数名为del_noise，可以通过肉眼看到很明显的区别，能更轻松地通过肉眼识别验证码。经过去噪声，模型有了从0到1的飞跃，实现了60%的验证码识别准确率，再经过调整参数（学习率、Dropout比例、训练次数、优化器选择等）实现测试集80%以上准确率、验证集准确率75%以上。

5、自动标注及人工校验。在模型准确率80%的基础上，重复第一步获取1000张验证码，通过以上模型进行自动标注验证码，人工校验。

6、模型更新。在第五步的基础上，将所有图片数据灌入模型，通过重复第四步，使得模型准确率达到90%左右。

7、迭代更新，重复上两步，最终模型止步于验证集96%准确率，图片数量达6000！

8、模型应用。基于96%准确率及验证码失败次数5次的基础上，设置尝试登录3次，三次成功率99.99%，再失败转入手工登录，这部分代码与公司紧密相关未上传，网商找找简单爬虫，了解下request机制、抓包即可随意处理。
