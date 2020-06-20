# EE208_M3DV
运行方式
===========
1.环境配置： tensorflow2,keras,python3.7,numpy(以Kaggle kernel上GPU环境为准）
---------
2.路径修改： test.py 378~381行 TRAIN_INFO/TRAIN_NODULE_PATH分别对应训练集的.csv标签文件与.npz数据文件（测试文件类似）
---------
3.命令输出： python test.py (或 python3 test.py)
---------
test.py 架构
===========
1.模型需要的基本函数：对准确率，召回率，损失函数等的计算与定义（precision.recall,Diceloss...)
---------
2.Densesharp模型的组成架构：压缩层，卷积层等构建函数，以及模型的整体构建与编译函数 （Dense_block,conv_block，get_model，get_compiled...）
---------
3.数据读取与预处理：读取数据文件，并对读取的数据通过旋转，随机选取中心信息等方式进行预处理（load_TrainDataset,load_TestDataset,Transform..)
---------
4.具体模型训练,编译:main函数，返回最终模型
---------
5.结果预测并生成.csv文件：结果预测（model.predict()），生成.csv文件 （load_result）
---------
