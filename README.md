# ST-SSL: Spatio-Temporal Self-Supervised Learning for Traffic Prediction 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nycbike1)](https://paperswithcode.com/sota/traffic-prediction-on-nycbike1?p=spatio-temporal-self-supervised-learning-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nycbike2)](https://paperswithcode.com/sota/traffic-prediction-on-nycbike2?p=spatio-temporal-self-supervised-learning-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nyctaxi)](https://paperswithcode.com/sota/traffic-prediction-on-nyctaxi?p=spatio-temporal-self-supervised-learning-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-bjtaxi)](https://paperswithcode.com/sota/traffic-prediction-on-bjtaxi?p=spatio-temporal-self-supervised-learning-for)

This is a Pytorch implementation of ST-SSL in the following paper: 

* J. Ji, J. Wang, C. Huang, et al. "Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction". in Thirty-Seventh AAAI Conference on Artificial Intelligence, 2023. 

The homepage of J. Ji is available at [here](https://echo-ji.github.io/academicpages/).

## Requirement

We build this project by Python 3.8 with the following packages: 
```
numpy==1.21.2
pandas==1.3.5
PyYAML==6.0
torch==1.10.1
```

## Model training and Evaluation

If the environment is ready, please run the following commands to train model on the specific dataset from `{NYCBike1, NYCBike2, NYCTaxi, BJTaxi}`.
```bash
>> cd ST-SSL
>> ./runme 0 NYCBike1   # 0 gives the gpu id
```

This repo contains the NYCBike1 data. If you are interested in other datasets, please download from [ST-SSL_Dataset](https://github.com/Echo-Ji/ST-SSL_Dataset).

## Cite

If you find the paper useful, please cite as following:

```
@inproceedings{ji2023spatio, 
  title={Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction}, 
  author={Ji, Jiahao and Wang, Jingyuan and Huang, Chao and Wu, Junjie and Xu, Boren and Wu, Zhenhe and Zhang, Junbo and Zheng, Yu}, 
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  year={2023}
}
```

## 初始代码地址
```
https://github.com/Echo-Ji/ST-SSL
```

## 自己修改增加的预训练的执行顺序
```
先执行 python pre_train.py 
随后执行 python new_trainer.py
```


## debug
```
在更换数据集过程中出现问题，dataloader加载npz文件时出现错误，在np.load函数中添加（encoding='bytes', allow_pickle=True）
但是 再加入上述属性后，仍然出现问题，OSError,查阅博客后提示需要下载git lfs
最后确认了一下 是因为数据集只能使用git lfs clone进行下载，并且通过git lfs进行管理控制，才能正常使用
```

## 