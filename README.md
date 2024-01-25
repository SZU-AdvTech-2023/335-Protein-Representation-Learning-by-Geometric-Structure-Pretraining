1.GearNet.py文件提供了GearNet端到端表征的功能，需要修改的路径：模型状态字典路径、PDB文件路径、保存路径
2.ESMFold_predict_cpu/gpu.py可以从蛋白质序列预测其结构，使用cpu或者gpu
3.mc_gearnet_edge.pth是gearnet的状态字典
4.环境参考官方仓库：
GearNet: https://github.com/DeepGraphLearning/GearNet
ESMFold: https://github.com/facebookresearch/esm