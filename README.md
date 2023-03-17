## 模型思路

在比赛前期提交的时候，发现线上线下分数差异很大。我们认为线上测试集中存在两种数据，一是见过的数据（与训练集类似的数据），二是没见过的数据（与训练集分布距离大的数据）。

针对这个特点，我们的建模思路为：开发两个类型的模型，第一个类型的模型用于解决见过的数据，这个类型的模型以线下训练损失越小越好；第二个类型的模型用于解决没见过的数据，这个类型的模型要泛化能力越强越好。

### 第一类型的模型

这个类型的模型有三个，分别是modelDesign中的encoder1、encoder2、encoder3。这三个的结构都是类似的。都采用背景法建模，背景法的原理在于：从零还原数据难度大，但是从相似数据中还原数据难度小，具体的模型结构如下图所示：  
<img src="https://github.com/liangoy/csi_compression_oppo/blob/main/encoder.jpg" width="50%" height="50%" />  
图表 1编码器结构  
<img src="https://github.com/liangoy/csi_compression_oppo/blob/main/decoder.jpg" width="50%" height="50%" />  
图表 2解码器结构  

下面简单介绍一下encoder1、encoder2与encoder3的区别

#### Encoder1

Encoder1的背景集合为w.mat，w.mat共有样本1000份，所以encoder1的背景集合的数量为1000。encoder1根据csi矩阵中相邻频率间的数据具有相关性的特点，按频域选择背景。具体来讲，encoder1分成两个部分取背景，前7个频率作为一个整体选取背景，后6个频率作为一个整体选取背景。每个背景的索引需要10bit来储存，所以encoder1共用了20bit储存背景索引，余下的bit用于储存神经网络的压缩结果。

#### Encoder2

Encoder2的背景集合与encoder1一样。Encoder2直接选用与原数据最接近的背景作为背景。所以encoder2使用了10bit储存背景索引，余下的bit用于储存神经网络的压缩结果。

#### Encoder3

Encoder3的背景集合来自H.mat。H.mat有1000个样本，每个样本有52个频率，有4条接收天线，8条发送天线。我们以将8条发送天线作为背景基本单元，所以得到了1000*52*4个背景，作为encoder3的背景集合。所以encoder3用了18bit（log2(1000*52*4)）记录背景索引，余下的比特用于储存神经网络的压缩结果。

### 第二类型的模型

这个类型的模型有两个，为encoder0和encode4。

encoder0与encoder1结构类似，区别在于encoder0没有背景相关的结构以及在训练的时候，为了使得encoder0具有强的泛化能力，我们在训练的时候加入一个额外的损失即样本的还原数据间的余弦相似度，这个余弦相似度越小表明还样本原数据越散，泛化能力越强。

Encoder4不需要训练，其压缩流程为，求出13\*8的csi复矩阵的最大特征向量，这个最大特征向量维度为8，需要用16个浮点数记录。我们对各个浮点数进行分箱达到压缩的目的。

### 求最优值

在推理的时候，如果使用跟训练过程一致的流程即:

1. 对于原数据，我们记为y_
2. 压缩，z=encoder(y_)
3. 解压缩，y=encoder(z)

在encoder与decoder确定的情况下，这套流程得到的压缩结果z并不是最优的。最优的z应该为：
```math
\mathop{\arg\min}\limits_z{\ distance}(y\_,decoder(z))
```

要计算出这样的z，我们需要穷举所有的z即可。

但穷举的计算速度太慢，所以我们采取了一些措施，可以在短时间内求得接近最优解的z。具体步骤如下：

1.  用 $encoder(y\\_)$初始化z

2.  根据distance(y_,decoder(z))关于z的梯度更z（z取值范围为0到1），迭代若干轮

3.  计算z的确定程度： $c=\\left| z-0.5 \\right|$，c中的元素越大，代表其对应的z元素的确定度越高。

4.  按照确定程度c对z中的元素重新排列（升序）,结果记为 $z_{sorted}$。

5.  对 $z_{sorted}$ ，使用beam search求解得到z最优解的排列后的近似解 ${z_{sorted}^\*}$
6.  对 ${{z_{sorted}}^\*}$逆排列，得到z最优解的近似解 ${z^\*}$,  ${z^\*}$ 即为最终压缩结果。

我们所有的模型均在推理时，均采用上述步骤得到压缩结果。

## 数据增强方案

我们的数据增强方案很简单，代码如下：
```python3
class Dataset(torch.utils.data.Dataset):
    '''
    训练数据生成器，生成数据时，包含了一些数据增强的操作
    '''
    def __init__(self,H):
        super(Dataset, self).__init__()
        self.H=H
    def __getitem__(self,i):
        i=np.random.randint(1000)
        i2=np.random.randint(4)
        index=np.arange(0,52,4)+np.random.randint(0,4,13)
        n=self.H[i,:,:,index].conj().transpose(0,2,1)#在频域上进行数据增强
        r=np.random.randn(4)+np.random.randn(4)*1j
        n=n*r#数据增强，对4个接收天线进行放缩
        n=np.linalg.svd(n)[0][:,:,0]
        n=np.stack([n.real,n.imag],-1)
        n=n.reshape([n.shape[0],-1])
        return n.astype(np.float32)
    def __len__(self):
        return 10**10
```

数据增强主要分为两个部分，一是在频域上进行采样，具体做法为将52个频率分成按间隔4分成13份，每份有4个数据 ，随机抽取一个作为这个份数据的代表；一是对4个接收天线进行放缩，用于模仿不同接收端。

## 模型结构

考虑到数据频域间具有高度的局部相关性，我们在transformer中加入了沿着频域卷积的1d cnn，即采用cnn与self-attention交错堆叠的方式组成transformer。这个结构比原始的transformer结构表现稍好，但不构成主要提分点。

## 总结

本方案的主要贡献有三个，如下所示：

1. 根据训练数据与线上数据分布差异大的情况，针对性地提出了两类模型，第一是解决见过的数据的模型，第二是解决没见过的数据的模型。使得最终的模型既有准确度又有泛化能力

2. 将压缩问题转化成离散最优化的问题，并提出快速求离散最优值得方案

3. 根据数据在频域上具有高度局部相关性的特点，在transformer上加入卷积，提升了模型表现

## 代码文件简介

* train01.py encoder0任务1的训练代码
* train02.py encoder0任务2的训练代码
* train11.py encoder1任务1的训练代码
* train12.py encoder1任务2的训练代码
* train21.py encoder2任务1的训练代码
* train22.py encoder2任务2的训练代码
* train31.py encoder3任务1的训练代码
* train32.py encoder3任务2的训练代码
* merge.py 将训练结果合并起来，生成最终模型文件
* modelDesign.py 工具函数以及模型结构

## 训练流程

依次调用train01.py,train02.py,..,train32.py完成各个子模型的训练，然后再调用merge.py整理训练结果生成模型文件
