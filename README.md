## Requirement

We build this project by Python 3.8 with the following packages: 
```
numpy==1.21.2
pandas==1.3.5
PyYAML==6.0
torch==1.10.1
```

## 初始代码地址
```
https://github.com/Echo-Ji/ST-SSL
```

## 自己修改增加的预训练的执行顺序
```
执行全程：python UMSST.py -mode pre_train --config_filename configs/NYCBike1.yaml  -model STSSL --step_1 0 --step_2 0 --set_attenuation True
执行下游任务：python UMSST.py -mode base --config_filename configs/NYCBike1.yaml  -model STSSL --set_attenuation False
```


## debug
```
在更换数据集过程中出现问题，dataloader加载npz文件时出现错误，在np.load函数中添加（encoding='bytes', allow_pickle=True）
但是 再加入上述属性后，仍然出现问题，OSError,查阅博客后提示需要下载git lfs
最后确认了一下 是因为数据集只能使用git lfs clone进行下载，并且通过git lfs进行管理控制，才能正常使用
```

