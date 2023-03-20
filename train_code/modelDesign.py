import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from hashlib import md5
import math
import copy
import random
'''
seed
'''
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    
set_seed(0)

def _cal_best2(fn,inp,y_,dims=10,top_n=1):
    _inp=inp
    out_list=list()
    score_list=list()
    for i in range(inp.shape[1]):
        inp=_inp[:,i]
        
        if i==0:
            index=(inp-0.5).abs().argsort(-1)
            M=torch.diag(torch.ones(inp.shape[-1],device=inp.device,dtype=torch.float32))
            T=M[index]
        inp_t=torch.einsum('bm,bnm->bn',inp,T)

        i=int2b(torch.arange(2**dims,device=inp.device,dtype=inp.dtype),dims)
        zeros=torch.zeros([2**dims,inp.shape[-1]-dims ],device=inp.device,dtype=inp.dtype)#!!
        i=torch.cat([ i,zeros ],-1)

        zeros=torch.zeros([inp.shape[0],dims],device=inp.device,dtype=inp.dtype)
        out=torch.cat([zeros,inp_t[:,dims:]],-1)

        out=out+i.unsqueeze(1)#[2**dims,b,d]

        out=torch.einsum('abm,bmn->abn',out,T)

        out=out.reshape([-1,out.shape[-1]])

        y=fn(out).reshape([-1,y_.shape[0],13,16])
        score,index=complex_corr(y,y_).mean(-1).sort(0)
        index=index[-top_n:].T
        score=score[-top_n:].T
        b=int2b(index,dims).type(inp.dtype)#[batch_size,top_n,d]
        inp_t=inp_t[:,dims:].unsqueeze(1).repeat(1,top_n,1)
        out=torch.cat([b,inp_t],-1)
        out=torch.einsum('bam,bmn->ban',out,T)
        
        out_list.append(out)
        score_list.append(score)
    out=torch.cat(out_list,1)
    score=torch.cat(score_list,1)
    index=score.argsort(-1)[:,-top_n:]
    out=batch_gather(out,index)
    return out

def cal_best2(fn,inp,y_,dims=10,top_n=1):
    '''
    _cal_best2(fn,inp,y_,dims,top_n=1)=agrmax{z}( distance(y_,inp*(1-I(inp,dims))+z*I(inp,dims)) )
    I为指示函数，指出inp中哪些位置需要更新
    distance为余弦距离
    inp为初始状态，其元素值介于0到1，在信道压缩的场景中，inp的形状为[batch_size,beam_serch_size,hidden_size]
    dims为需要替换的维度的数量，决定了I，具体来说，优先将inp中不确定（靠近0.5）的元素进行替换，替换的数量为dims
    top_n为beam_serch_size，保留前top_n个最好结果
    '''
    with torch.no_grad():
        if dims<10:
            out=_cal_best2(fn,inp,y_,dims,top_n)
        else:
            i=len(inp)//2
            out1=_cal_best2(fn,inp[:i],y_[:i],dims,top_n)
            out2=_cal_best2(fn,inp[i:],y_[i:],dims,top_n)
            out=torch.cat([out1,out2],0)
    return out

def _cal_best(fn,inp,y_,start=None,end=None,top_n=1):
    _inp=inp
    out_list=list()
    score_list=list()
    for i in range(_inp.shape[1]):
        inp=_inp[:,i]
        x=inp
        inp_start=inp[:,:start]
        inp_end=inp[:,end:]

        dims=end-start


        i=int2b(torch.arange(2**dims,device=inp.device,dtype=inp.dtype),dims)
        zeros=torch.zeros([2**dims,inp.shape[-1] ],device=inp.device,dtype=inp.dtype)#!!
        i=torch.cat([   zeros[:,:start],i,zeros[:,end:]   ],-1)
        zeros=torch.zeros([inp.shape[0],dims],device=inp.device,dtype=inp.dtype)
        out=torch.cat([  inp_start,zeros,inp_end   ],-1)

        out=out+i.unsqueeze(1)

        out=out.reshape([-1,out.shape[-1]])

        y=fn(out).reshape([-1,y_.shape[0],13,16])

        score,index=complex_corr(y,y_).mean(-1).sort(0)
        index=index[-top_n:].T
        score=score[-top_n:].T
        b=int2b(index,dims).type(inp.dtype)#[batch_size,top_n,d]
        inp_start=inp_start.unsqueeze(1).repeat(1,top_n,1)
        inp_end=inp_end.unsqueeze(1).repeat(1,top_n,1)
        out=torch.cat([inp_start,b,inp_end],-1)
        
        out_list.append(out)
        score_list.append(score)
    out=torch.cat(out_list,1)
    score=torch.cat(score_list,1)
    index=score.argsort(-1)[:,-top_n:]
    out=batch_gather(out,index)
    return out

def cal_best(fn,inp,y_,start=None,end=None,top_n=1):
    '''
    _cal_best(fn,inp,y_,start=None,end=None,top_n=1)=agrmax{z}( distance(y_,inp*(1-I(start,end))+z*I(start,end)) )
    I为指示函数，指出inp中哪些位置需要更新
    distance为余弦距离
    inp为初始状态，其元素值介于0到1，在信道压缩的场景中，inp的形状为[batch_size,beam_serch_size,hidden_size]
    start：从第几维开始替换
    end：到第几维结束替换
    top_n为beam_serch_size，保留前top_n个最好结果
    '''
    with torch.no_grad():
        if end-start<10:
            out=_cal_best(fn,inp,y_,start,end,top_n)
        else:
            i=len(inp)//2
            out1=_cal_best(fn,inp[:i],y_[:i],start,end,top_n)
            out2=_cal_best(fn,inp[i:],y_[i:],start,end,top_n)
            out=torch.cat([out1,out2],0)
    return out

class Round(torch.autograd.Function):
    '''
    量化
    '''
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ParallelLayer(torch.autograd.Function):
    '''
    将输入复制n倍
    
    '''
    @staticmethod
    def forward(ctx, x,n=3):
        oup=x.repeat([n]+[1]*(len(x.shape)-1))
        ctx.n=n
        return oup

    @staticmethod
    def backward(ctx, grad_output):
        shape=list(grad_output.shape)
        grad_output=grad_output.reshape([ctx.n,shape[0]//ctx.n]+shape[1:])
        grad_output=grad_output.mean(0)
        return grad_output, None

def int2b(x,n=5):
    '''
    将int转化成01比特，n代表比特数
    
    '''
    if x.dtype==torch.float16 or x.dtype==torch.float32 or x.dtype==torch.float64:
        x=torch.round(x)
    x=x.unsqueeze(-1)
    a=2**torch.arange(n,device=x.device,dtype=x.dtype)
    b=x//a%2
    return b
def b2int(x,n=5):
    '''
    将01比特转化成int，n代表比特数
    '''
    if x.dtype==torch.float16 or x.dtype==torch.float32 or x.dtype==torch.float64:
        x=torch.round(x)
    a=2**torch.arange(n,device=x.device,dtype=x.dtype)
    x=(x*a).sum(-1)
    return x

def layer_norm(x):
    return torch.layer_norm(x,x.shape[-1:])

def cal_norm(x):
    '''
    计算x最后一阶的l2范数
    x是用浮点数表示的复数张良，
    第2n个数表示第n个数的实部，第2n+1个数表示第n个数的虚部
    
    '''
    x=x.T
    a=x[::2]
    b=x[1::2]
    norm=(a.square()+b.square()).sum(0).sqrt().T
    return norm

def do_norm(x):
    '''
    将x最后一阶转成单位向量
    x是用浮点数表示的复数张量，
    第2n个数表示第n个数的实部，第2n+1个数表示第n个数的虚部
    '''
    x=x.T
    a=x[::2]
    b=x[1::2]
    norm=(a.square()+b.square()).sum(0).sqrt()
    x=(x/norm).T
    return x

def complex_corr(a,b,norm_a=False,norm_b=False):
    '''
    计算a,b的相似度，a、b是用浮点数表示的复数张量，第2n个数表示第n个数的实部，第2n+1个数表示第n个数的虚部。
    norm_a：是否需要将a转成单位向量
    norm_b：是否需要将b转成单位向量
    '''
    
    
    if norm_a:
        a=do_norm(a)
    if norm_b:
        b=do_norm(b)
        
    a=a.T
    ai=a[::2].T
    aj=a[1::2].T

    b=b.T
    bi=b[::2].T
    bj=b[1::2].T    
    
    i=(ai*bi+aj*bj).sum(-1)
    j=(ai*bj-aj*bi).sum(-1)
    c=i*i+j*j
    return c


def cal_nn(y,x):
    '''
    对x乘上一个复数，使得x与y的欧式距离最近，cal_nn返回x与这个复数的乘积。
    第2n个数表示第n个数的实部，第2n+1个数表示第n个数的虚部
    '''
    shape=tuple(x.shape)
    x=x.unsqueeze(0).transpose(0,-1)
    y=y.unsqueeze(0).transpose(0,-1)
    xi,xj=x[::2],x[1::2]
    yi,yj=y[::2],y[1::2]
    z=(xi**2+xj**2).sum(0)
    a=(xi*yi+xj*yj).sum(0)/z
    b=(xi*yj-xj*yi).sum(0)/z
    real=(a*xi-b*xj).transpose(0,-1)[0]
    imag=(b*xi+a*xj).transpose(0,-1)[0]
    x=torch.stack([real,imag],-1).reshape(shape)
    return x

def swish(x):
    return x * torch.sigmoid(x)

class Mixer(torch.nn.Module):
    def __init__(self,dims1,dims2,drop_rate=0.2):
        super(Mixer,self).__init__()
        self.fc1=torch.nn.Linear(dims1,dims1)
        self.fc2=torch.nn.Linear(dims2,dims2)
        self.drop_rate=drop_rate
    def forward(self,x):
        out=x
        
        inp=out
        out=torch.nn.functional.dropout(out,p=self.drop_rate,training=self.training)
        out=layer_norm( inp*2+swish(self.fc2(out.transpose(-1,-2))).transpose(-1,-2)   )
        
        inp=out
        out=torch.nn.functional.dropout(out,p=self.drop_rate,training=self.training)
        out=layer_norm(inp*2+swish(self.fc1(out)))
        
        return out

class Cnn(torch.nn.Module):
    '''
    一维卷积
    '''
    def __init__(self,dims,drop_rate=0.2):
        super(Cnn,self).__init__()
        self.cnn=torch.nn.Conv1d(dims,dims,3,1,1)
        self.drop_rate=drop_rate
    def forward(self,x):
        out=x
        out=torch.nn.functional.dropout(out,p=self.drop_rate,training=self.training)
        out=self.cnn(out.transpose(-1,-2)).transpose(-1,-2)
        out=swish(out)
        out=layer_norm(x*2+out)
        return out

class CnnAttention(torch.nn.Module):
    '''
    区别于普通的Attention，CnnAttention通过Cnn计算出qkv
    '''
    def __init__(self,dims,hd,heads=1,drop_rate=0.1):
        super(CnnAttention,self).__init__()
        self.cnn=torch.nn.Conv1d(dims,heads*hd*3,3,1,1)
        self.fcfc=torch.nn.Linear(hd*heads,dims)
        self.dims=dims
        self.hd=hd
        self.heads=heads
        self.drop_rate=drop_rate
    def forward(self,x):
        batch_size=x.shape[0]
        length=x.shape[1]
        x_dp=torch.nn.functional.dropout(x,p=self.drop_rate,training=self.training)
        qkv=swish(self.cnn(x_dp.transpose(-1,-2)))
        qkv=qkv.reshape([batch_size,self.heads,self.hd*3,length])
        q,k,v=torch.split(qkv,self.hd,-2)
        att=torch.softmax(torch.einsum('bhdw,bhdm->bhwm',q,k)/np.sqrt(length),-1)
        new_v=torch.einsum('bhwm,bhdm->bwhd',att,v).reshape([batch_size,length,-1])
        
        m=swish(self.fcfc(new_v))
        return layer_norm(x*2+m)

class Attention(torch.nn.Module):
    def __init__(self,dims,hd,heads=1,drop_rate=0.1):
        super(Attention,self).__init__()
        self.fc=torch.nn.Linear(dims,heads*hd*3)
        self.fcfc=torch.nn.Linear(hd*heads,dims)
        self.dims=dims
        self.hd=hd
        self.heads=heads
        self.drop_rate=drop_rate
    def forward(self,x):
        batch_size=x.shape[0]
        length=x.shape[1]
        x_dp=torch.nn.functional.dropout(x,p=self.drop_rate,training=self.training)
        qkv=swish(self.fc(x_dp))
        qkv=qkv.reshape([batch_size,length,self.heads,self.hd*3])
        q,k,v=torch.split(qkv,self.hd,-1)
        att=torch.softmax(torch.einsum('bwhd,bmhd->bhwm',q,k)/np.sqrt(length),-1)
        new_v=torch.einsum('bhwm,bmhd->bwhd',att,v).reshape([batch_size,length,-1])
        
        m=swish(self.fcfc(new_v))
        return layer_norm(x+m)

def batch_gather(x, index):
    shape = list(x.shape)
    x_flat = x.reshape([-1]+shape[2:])
    index_flat = torch.arange(shape[0],device=index.device,dtype=index.dtype)*shape[1]
    index_flat = (index_flat+index.T).T
    out = x_flat[index_flat]
    return out

def nnv(x):
    '''
    求复张量x的最大特征向量
    第2n个数表示第n个数的实部，第2n+1个数表示第n个数的虚部
    '''
    _x=x
    x=(x.T[::2]+x.T[1::2]*1j).T.transpose(-1,-2)
    x=x.detach().cpu().numpy()
    u,s,vh=np.linalg.svd(x)
    u=u.T[0].T
    u=torch.tensor(u,device=_x.device)
    u=torch.stack([u.real,u.imag],-1).reshape(list(_x.shape[:-2])+[-1])
    return u

def pca(x,top_n=1):
    _x=x
    x=(x.T[::2]+x.T[1::2]*1j).T.transpose(-1,-2)
    x=x.detach().cpu().numpy()
    u,s,vh=np.linalg.svd(x)
    u=u.T[:top_n].T
    u=torch.tensor(u,device=_x.device)
    u=torch.stack([u.real,u.imag],-1).reshape(list(u.shape[:-1])+[-1])
    return u
def nnv_torch(x):
    _x=x
    x=(x.T[::2]+x.T[1::2]*1j).T.transpose(-1,-2)
    u,s,vh=torch.linalg.svd(x)
    u=u.T[0].T
    u=torch.stack([u.real,u.imag],-1).reshape(list(_x.shape[:-2])+[-1])
    return u

def sort(x):
    '''
    根据x特征向量每个元素的大小，对x进行排列
    '''
    v=nnv(x)
    r,i=x.T[::2].T,x.T[1::2].T
    v_norm=v.T[::2].T.square()+v.T[1::2].T.square()
    E=torch.diag(torch.ones(v_norm.shape[-1],dtype=v.dtype,device=v.device))
    E=E[v_norm.argsort(-1)].unsqueeze(1)
    rr=torch.matmul(E,r.unsqueeze(-1))
    ii=torch.matmul(E,i.unsqueeze(-1))
    x_sorted=torch.cat([rr,ii],-1).reshape_as(x)
    return x_sorted

def best(c,a,b,return_w=False):
    '''
    返回a*w+b
    w=argmax{w}( distance(c,a*w+b) )
    distance为余弦相似度
    '''
    shape=list(c.shape)
    a=(a.T[::2]+a.T[1::2]*1j).T
    b=(b.T[::2]+b.T[1::2]*1j).T
    c=(c.T[::2]+c.T[1::2]*1j).T
    aa=(a*a).sum(-1)
    ab=(a*b).sum(-1)
    ac=(a*c).sum(-1)
    bb=(b*b).sum(-1)
    bc=(b*c).sum(-1)
    w=(ab*bc-ac*bb)/(ab*ac-aa*bc)
    x=(a.T*w.T+b.T).T
    x=do_norm(torch.stack([x.real,x.imag],-1).reshape(list(x.shape[:-1])+[-1] ))
    if return_w:
        w=torch.stack([w.real,w.imag],-1).reshape(list(x.shape[:-1])+[2] )
        return x,w
    else:
        return x
def complex_dot(a,b):
    '''
    复数相乘
    '''
    ar=a.T[::2].T
    ai=a.T[1::2].T
    br=b.T[::2].T
    bi=b.T[1::2].T
    r=ar*br-ai*bi
    i=ar*bi+ai*br
    x=torch.stack([r,i],-1).reshape(list(r.shape[:-1])+[-1])
    return x
#=======================================================================================================================
#=======================================================================================================================
def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1*num2))
    return cos

def cal_score(w_true,w_pre,NUM_SAMPLES,NUM_SUBBAND):
    img_total = 16
    num_sample_subband = NUM_SAMPLES * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_subband, img_total])#[b*13,8*2]
    W_pre = np.reshape(w_pre, [num_sample_subband, img_total])
    W_true2 = W_true[0:num_sample_subband, 0:int(img_total):2] + 1j*W_true[0:num_sample_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_subband, 0:int(img_total):2] + 1j*W_pre[0:num_sample_subband, 1:int(img_total):2]
    score_cos = 0
    for i in range(num_sample_subband):
        W_true2_sample = W_true2[i:i+1,]
        W_pre2_sample = W_pre2[i:i+1,]
        score_tmp = cos_sim(W_true2_sample,W_pre2_sample)
        score_cos = score_cos + abs(score_tmp)*abs(score_tmp)
    score_cos = score_cos/num_sample_subband
    return score_cos
#=======================================================================================================================
#=======================================================================================================================
# Data Loader Class Defining
class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]


'''
for training
'''
class Saver:
    def __init__(self,model=None,path=None):
        self._r=str(np.random.randint(10**9))
        self.path=path+'_'+self._r
        self.loss=10**9
        self.model=model
        self.best_iter=0
        self.n_iter=0
    def update(self,loss):
        self.n_iter+=1
        loss=float(loss)
        if loss<self.loss:
            self.save(self.path)
            self.loss=loss
            self.best_iter=self.n_iter
    def save_best(self,path):
        with open(self.path,'rb') as f:
            b=f.read()
        with open(path,'wb')as f:
            f.write(b)
    def load_best(self):
        self.load(self.path)
    def save(self,path=None):
        torch.save({'state_dict':self.model.state_dict()},path)
    def load(self,path=None):
        self.model.load_state_dict(torch.load(path)['state_dict'])

'''
model design
'''

class _Encoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(_Encoder,self).__init__()
        self.background=torch.nn.Parameter(do_norm(torch.randn(1024,13,16)),requires_grad=False)
        self.fc1=torch.nn.Linear(32,16)
        lis=list()
        for i in range(3):
            lis.append(Cnn(16))
            lis.append(Attention(16,16,3))
        self.attentions=torch.nn.Sequential(*lis)
        self.fc2=torch.nn.Linear(13*16,8)
        self.PE=torch.nn.Parameter(torch.randn([13,16])/10)
    def forward(self,x):
        x=x.reshape([-1,13,16]).type(torch.float32)
        background=do_norm(self.background)
        loss=-complex_corr(x.unsqueeze(1),background)
        x_bg_list=list()
        index_list=list()
        start=0
        for _i,i in enumerate([7,6]):
            end=start+i
            index=loss[:,:,start:end].mean(-1).argsort(-1)
            if self.training:
                r=torch.randint(1,3,x.shape[:1],device=x.device,dtype=torch.int64)
                index=batch_gather(index,r)
            else:
                index=index[:,0]
            index_list.append(index)
            x_bg_list.append(background[index,start:end])
            start=end
        x_bg=torch.cat(x_bg_list,1)      
        out=do_norm(cal_nn(x_bg,x))
        
        
        out=torch.cat([out-x_bg,out],-1)
        
        out=layer_norm(self.fc1(out)+self.PE)
        out=self.attentions(out)
        
        
        out=out.reshape([x.shape[0],-1])
        out=self.fc2(out)
        out=torch.sigmoid(out/10)
        out=layer_norm(out.T).T
        out=torch.sigmoid(out)
        I=torch.stack(index_list,-1)
        b=int2b(I,10).type(torch.float32).reshape([x.shape[0],-1])
        out=torch.cat([b,out],-1)
        return out

class Decoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(Decoder,self).__init__()
        self.fc0=torch.nn.Linear(16,16)
        self.fc2=torch.nn.Linear(8,13*16)
        self.PE=torch.nn.Parameter(torch.randn([13,16])/10)
        lis=list()
        for i in range(3):
            lis.append(Cnn(16))
            lis.append(Attention(16,16,3))
        self.attentions=torch.nn.Sequential(*lis)
    def forward(self,x):
        x_bg_list=list()
        index_list=b2int(x[:,:20].reshape([len(x),2,10]),10).T.type(torch.int64)
        background=do_norm(self.background)
        start=0
        for _i,i in enumerate([7,6]):
            end=start+i
            index=index_list[_i]
            x_bg=background[index,start:end]
            x_bg_list.append(x_bg)
            start=end
        x_bg=torch.cat(x_bg_list,1)
        out=x[:,20:28]
        
        if not self.training:
            out=layer_norm(out//0.50001)
        else:
            tun=(torch.rand_like(out)>(1-0.2)).type(out.dtype)
            out=(layer_norm((torch.rand_like(out)>0.5).type(out.dtype)) )*tun+layer_norm(out)*(1-tun)
            
        out=self.fc2(out)
        out=layer_norm(out)
        out=out.reshape([x.shape[0],13,16])
#         out=out.repeat(1,13,1)
        out=layer_norm(out*0.1+x_bg+self.PE)
        out=self.attentions(out)
        out=self.fc0(out)*0.1+x_bg
        return do_norm(out)
        
    
class Encoder(torch.nn.Module):
    def __init__(self,d=512):
        super(Encoder, self).__init__()
        self.encoder=_Encoder().cuda()
        self.decoder=Decoder().cuda()
        self.decoder.background=self.encoder.background
    def forward(self,x):
        x=x.reshape([x.shape[0],13,16])
        z=self.encoder(x)
        y_=x
        
        z=z.unsqueeze(1)
        z=cal_best(self.decoder,z,y_,20,28,2)
        z=cal_best(self.decoder,z,y_,10,20,2)
        z=cal_best(self.decoder,z,y_,0,10,2)
        z=cal_best(self.decoder,z,y_,20,28,1)

        out=z[:,0]
        
        ones=torch.ones_like(out)[:,:1]
        zeros=torch.zeros_like(out)[:,:1]
        out=torch.cat([out,zeros,ones],-1)
        
        return out//0.50001

encoder1=Encoder()
# encoder1.load_state_dict(torch.load(f'project/encoder_{task}.pth.tar')['state_dict'])

class _Encoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(_Encoder,self).__init__()
        self.background=torch.nn.Parameter(do_norm(torch.randn(1024,13,16)),requires_grad=False)
        self.fc1=torch.nn.Linear(32,16)
        lis=list()
        for i in range(3):
            lis.append(Cnn(16))
            lis.append(CnnAttention(16,16,3))
        self.attentions=torch.nn.Sequential(*lis)
        self.fc2=torch.nn.Linear(13*16,18)
        self.PE=torch.nn.Parameter(torch.randn([13,16])/10)
    def forward(self,x):
        x=x.reshape([-1,13,16]).type(torch.float32)
        background=do_norm(self.background)
        loss=-complex_corr(x.unsqueeze(1),background)
        x_bg_list=list()
        index_list=list()
        start=0
        for _i,i in enumerate([13]):
            end=start+i
            index=loss[:,:,start:end].mean(-1).argsort(-1)
            if self.training:
                r=torch.randint(1,3,x.shape[:1],device=x.device,dtype=torch.int64)
                index=batch_gather(index,r)
            else:
                index=index[:,0]
            index_list.append(index)
            x_bg_list.append(background[index,start:end])
            start=end
        x_bg=torch.cat(x_bg_list,1)      
        out=do_norm(cal_nn(x_bg,x))
        
        
        out=torch.cat([out-x_bg,out],-1)
                
        out=layer_norm(self.fc1(out)+self.PE)
        out=self.attentions(out)
        
        
        out=out.reshape([x.shape[0],-1])
        out=self.fc2(out)
        out=torch.sigmoid(out/10)#half>0
        out=layer_norm(out.T).T#haft>0
        out=torch.sigmoid(out)
        I=torch.stack(index_list,-1)
        b=int2b(I,10).type(torch.float32).reshape([x.shape[0],-1])
        out=torch.cat([b,out],-1)
        return out

class Decoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(Decoder,self).__init__()
        self.fc0=torch.nn.Linear(16,16)
        self.fc1=torch.nn.Linear(32,16)
        self.fc2=torch.nn.Linear(18,13*16)
        self.PE=torch.nn.Parameter(torch.randn([13,16])/10)
        lis=list()
        for i in range(3):
            lis.append(Cnn(16))
            lis.append(CnnAttention(16,16,3))
        self.attentions=torch.nn.Sequential(*lis)
    def forward(self,x):
        x_bg_list=list()
        index=b2int(x[:,:10],10).type(torch.int64)
        x_bg=self.background[index]
        out=x[:,10:28]
        
        if not self.training:
            out=layer_norm(out//0.50001)
        else:
            tun=(torch.rand_like(out)>(1-0.2)).type(out.dtype)
            out=(layer_norm(torch.rand_like(out)) )*tun+layer_norm(out)*(1-tun)
            
        out=self.fc2(out)
        out=layer_norm(out)
        out=out.reshape([x.shape[0],13,16])
        out=self.fc1(torch.cat([out,out+x_bg],-1))
        out=layer_norm(out+self.PE)
        out=self.attentions(out)
        out=self.fc0(out)*0.1+x_bg
        return do_norm(out)

class Encoder(torch.nn.Module):
    def __init__(self,d=512):
        super(Encoder, self).__init__()
        self.encoder=_Encoder().cuda()
        self.decoder=Decoder().cuda()
        self.decoder.background=self.encoder.background
    def forward(self,x):
        x=x.reshape([x.shape[0],13,16])
        z=self.encoder(x)
        
        z=z.unsqueeze(1)
        y_=x
        z=cal_best2(self.decoder,z,y_,9,2)
        z=cal_best2(self.decoder,z,y_,9,2)
        z=cal_best(self.decoder,z,y_,0,10,1)
        z=cal_best(self.decoder,z,y_,10,19,1)
        z=cal_best(self.decoder,z,y_,19,28,1)

        out=z[:,0]
        
        ones=torch.ones_like(out)
        out=torch.cat([out,ones],-1)[:,:30]
        return out//0.50001
    
encoder2=Encoder()

class _Encoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(_Encoder,self).__init__()
        self.background=torch.nn.Parameter(torch.randn(1024,13,16),requires_grad=False)
        self.fc1=torch.nn.Linear(16,16)
        lis=list()
        cnn=Cnn(16)
        att=Attention(16,16,3)
        for i in range(3):
            lis.append(cnn)
            lis.append(att)
        self.attentions=torch.nn.Sequential(*lis)
        self.fc2=torch.nn.Linear(13*16,28)
        self.PE=torch.nn.Parameter(torch.randn([13,16])/10)
    def forward(self,x):
        x=x.reshape([-1,13,16]).type(torch.float32)        
        out=x
        ones=do_norm(torch.ones_like(out))
        out=do_norm(cal_nn(ones,out))
        
        out=layer_norm(self.fc1(out)+self.PE)
        out=self.attentions(out)
        
        out=out.reshape([x.shape[0],-1])
        out=torch.nn.functional.dropout(out,p=0.2,training=self.training)
        out=self.fc2(out)
        out=torch.sigmoid(out/10)#half>0
        out=layer_norm(out.T).T#haft>0
        out=torch.sigmoid(out)
        return out

class Decoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(Decoder,self).__init__()
        self.fc0=torch.nn.Linear(16,16)
        self.fc2=torch.nn.Linear(28,13*16)
        self.PE=torch.nn.Parameter(torch.randn([13,16])/10)
        lis=list()
        cnn=Cnn(16)
        att=Attention(16,16,3)
        for i in range(3):
            lis.append(cnn)
            lis.append(att)
        self.attentions=torch.nn.Sequential(*lis)

    def forward(self,x):
        out=x[:,:28]
        
        if not self.training:
            out=layer_norm(out//0.50001)
        else:
            tun=(torch.rand_like(out)>(1-0.2)).type(out.dtype)
            out=(torch.randint_like(out,0,2)*2-1 )*tun+layer_norm(out)*(1-tun)
        
        out=self.fc2(out)
        out=layer_norm(out)
        out=out.reshape([x.shape[0],13,16])
        out=layer_norm(out+self.PE)
        out=self.attentions(out)
        out=self.fc0(out)
        
        return do_norm(out)
    
class Encoder(torch.nn.Module):
    def __init__(self,d=512):
        super(Encoder, self).__init__()
        self.encoder=_Encoder().cuda()
        self.decoder=Decoder().cuda()
        self.decoder.background=self.encoder.background
    def forward(self,x,n_iter=100):
        decoder_training=self.decoder.training
        x=x.reshape([-1,13,16])
        _feature=self.encoder(x).detach()
        p=_feature[:,:28]
        logit=torch.log(p/(1-p+1e-7)+1e-7)
        w=torch.nn.Parameter(logit)
        optimizer=torch.optim.Adam([w],lr=0.1)
        if not self.decoder.training:
            self.decoder.train()
        lr=float(optimizer.param_groups[0]['lr'])
        lis=list()
        with torch.enable_grad():
            for i in range(n_iter):
                optimizer.param_groups[0]['lr']=min(lr/(n_iter*0.9)*(i+1),lr)
                out=w
                out=torch.sigmoid(out)
                y=self.decoder(ParallelLayer.apply(out))
                score=complex_corr(y,ParallelLayer.apply(x).detach()).mean()
                loss=-score
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_([w], clip_value=0.001)
                optimizer.step()
                if i>(n_iter*0.9):
                    lis.append(out)
        out=torch.stack(lis,0).mean(0)
        
        if self.decoder.training!=decoder_training:
            if self.decoder.training:
                self.decoder.eval()
            else:
                self.decoder.train() 
        
        y_=x
        z=out
        z=z.unsqueeze(1)
        zz=z
        z=torch.clip(z,0.05,0.95)
        z=cal_best2(self.decoder,z,y_,10,5)
        z=cal_best2(self.decoder,z,y_,9,4)
        z=cal_best2(self.decoder,z,y_,9,3)
        z=(z+zz)/2
        z=cal_best2(self.decoder,z,y_,9,1)
        z=z[:,0]
#         for i in range(2):
#             z=cal_best(self.decoder,z,y_,19,29)
#             z=cal_best(self.decoder,z,y_,9,19)
#             z=cal_best(self.decoder,z,y_,0,9)
        out=z
        
        zeros=torch.zeros_like(out)
        out=torch.cat([out,zeros],-1)[:,:30]
        return out//0.500001
    
encoder0=Encoder()
# encoder0.load_state_dict(torch.load(f'project6/encoder_{task}.pth.tar')['state_dict'])

class _Encoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(_Encoder,self).__init__()
        self.background=torch.nn.Parameter(do_norm(torch.randn(2**18,16)),requires_grad=False)
        self.fc1=torch.nn.Linear(32,16)
        lis=list()
        for i in range(3):
            lis.append(Cnn(16))
            lis.append(CnnAttention(16,8,6))
        self.attentions=torch.nn.Sequential(*lis)
        self.fc2=torch.nn.Linear(13*16,9)
        self.PE=torch.nn.Parameter(torch.randn([13,16])/10)
    def forward(self,x):
        x=x.reshape([-1,13,16]).type(torch.float32)
        if self.training:
            index_sample=torch.randint(0,2**18,[2**12],device=x.device)
            background=self.background[index_sample]
            index=complex_corr(x.unsqueeze(2),background).mean(1).argmax(-1)
            index=index_sample[index]
        else:
            index_list=list()
            background=self.background
            for i in range(0,len(x),20):
                _x=x[i:i+20]
                index=complex_corr(_x.unsqueeze(2),background).mean(1).argmax(-1)
                index_list.append(index)
            index=torch.cat(index_list,-1)
        I=index
        x_bg=self.background[I].unsqueeze(1).repeat(1,13,1)
        out=do_norm(cal_nn(x_bg,x))
        out=torch.cat([out-x_bg,out],-1)
        out=layer_norm(self.fc1(out)+self.PE)
        out=self.attentions(out)
        
        
        out=out.reshape([x.shape[0],-1])
        out=self.fc2(out)
        out=torch.sigmoid(out/10)#half>0
        out=layer_norm(out.T).T#haft>0
        out=torch.sigmoid(out)
        b=int2b(I,18).type(torch.float32).reshape([x.shape[0],-1])
        out=torch.cat([b,out],-1)
        return out

class Decoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(Decoder,self).__init__()
        self.fc0=torch.nn.Linear(16,16)
        self.fc1=torch.nn.Linear(32,16)
        self.fc2=torch.nn.Linear(9,13*16)
        self.PE=torch.nn.Parameter(torch.randn([13,16])/10)
        lis=list()
        for i in range(3):
            lis.append(Cnn(16))
            lis.append(CnnAttention(16,8,6))
        self.attentions=torch.nn.Sequential(*lis)
    def forward(self,x):
        x_bg_list=list()
        index=b2int(x[:,:18],18).type(torch.int64)
        x_bg=self.background[index].unsqueeze(1).repeat(1,13,1)
        out=x[:,18:27]
        
        if not self.training:
            out=layer_norm(out//0.50001)
        else:
            tun=(torch.rand_like(out)>(1-0.4)).type(out.dtype)
            out=(layer_norm((torch.rand_like(out)>0.5).type(out.dtype)) )*tun+layer_norm(out)*(1-tun)
            
        out=self.fc2(out)
        out=layer_norm(out)
        out=out.reshape([x.shape[0],13,16])
        out=self.fc1(torch.cat([out,x_bg],-1))
        out=layer_norm(out+self.PE)
        out=self.attentions(out)
        out=self.fc0(out)*0.1+x_bg
        return do_norm(out)
    
class Encoder(torch.nn.Module):
    def __init__(self,d=512):
        super(Encoder, self).__init__()
        self.encoder=_Encoder().cuda()
        self.decoder=Decoder().cuda()
        self.decoder.background=self.encoder.background
    def forward(self,x):
        x=x.reshape([x.shape[0],13,16])
        z=self.encoder(x)
        
        z=z.unsqueeze(1)
        y_=x
        z=cal_best(self.decoder,z,y_,18,27,2)
        z=cal_best(self.decoder,z,y_,9,18,2)
        z=cal_best(self.decoder,z,y_,0,9,2)
        z=cal_best(self.decoder,z,y_,18,27,1)

        out=z[:,0]
        
        ones=torch.ones_like(out)[:,:1]
        zeros=torch.zeros_like(out)[:,:1]
        out=torch.cat([out,zeros,ones,zeros],-1)
        return out//0.50001

encoder3=Encoder()
class _Encoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(_Encoder,self).__init__()
        self.background=None
    def forward(self,x):
        x=x.reshape([-1,13,16]).type(torch.float32)  
        v=nnv(x)
        index=(v[:,::2].square()+v[:,1::2].square()).argmax(-1)

        c=batch_gather(v.reshape(len(v),-1,2),index)
        c=torch.stack([c[:,0],-c[:,1]],-1)
        v=complex_dot(v,c)
        v=(v.T/v.abs().T.max(0)[0]).T

        flag=torch.nn.functional.one_hot(index)+torch.arange(8,dtype=index.dtype,device=index.device)/100
        ones=torch.diag(torch.ones(8,dtype=flag.dtype,device=flag.device))
        E=ones[flag.argsort(-1)]

        v_r=torch.matmul(E,v[:,::2].unsqueeze(-1)).squeeze(-1)[:,:7]
        v_i=torch.matmul(E,v[:,1::2].unsqueeze(-1)).squeeze(-1)[:,:7]

        v=torch.cat([v_r,v_i],-1)

        v1=torch.round((v[:,:4]+1)/2*3)
        v2=torch.round((v[:,4:]+1)/2*2)

        b0=int2b(index,3).type(torch.float32)
        b1=int2b(v1,2).reshape(len(v1),-1)
        v2=v2.type(torch.int64)
        s=(v2*3**torch.arange(10,dtype=v.dtype,device=v.device)).sum(-1)
        b2=int2b(s,16).type(torch.float32)
        z=torch.cat([b0,b1,b2],-1)
        out=z
        return out

class Decoder(torch.nn.Module):
    def __init__(self,d=30,hd=16):
        super(Decoder,self).__init__()
    def forward(self,x):
        z=x[:,:27]
        index=b2int(z[:,:3],3).type(torch.int64)


        v1=b2int(z[:,3:11].reshape(-1,4,2),2)
        v1=v1/3*2-1
        s=b2int(z[:,11:27].type(torch.int64),16).unsqueeze(-1)
        v2=s//3**torch.arange(10,dtype=s.dtype,device=s.device)%3
        v2=v2.type(torch.float32)
        v2=v2/2*2-1
        v=torch.cat([v1,v2],-1)

        v_r=torch.cat([v[:,:7],torch.ones([len(v),1],dtype=v.dtype,device=v.device)],-1)
        v_i=torch.cat([v[:,7:],torch.zeros([len(v),1],dtype=v.dtype,device=v.device)],-1)

        flag=torch.nn.functional.one_hot(index,8)+torch.arange(8,dtype=index.dtype,device=index.device)/100
        ones=torch.diag(torch.ones(8,dtype=flag.dtype,device=flag.device))
        E=ones[flag.argsort(-1)].transpose(-1,-2)

        v_r=torch.matmul(E,v_r.unsqueeze(-1)).squeeze(-1)
        v_i=torch.matmul(E,v_i.unsqueeze(-1)).squeeze(-1)

        v=torch.stack([v_r,v_i],-1).reshape([-1,16])
        v=do_norm(v+1e-7).unsqueeze(1).repeat(1,13,1)
        return v
    
class Encoder(torch.nn.Module):
    def __init__(self,d=512):
        super(Encoder, self).__init__()
        self.encoder=_Encoder().cuda()
        self.decoder=Decoder().cuda()
        self.decoder.background=self.encoder.background
    def forward(self,x):
        x=x.reshape([x.shape[0],13,16])
        z=self.encoder(x)
        z=z.unsqueeze(1)
        y_=x
        beam_n=27
        z=cal_best(self.decoder,z,y_,18,27,beam_n)
        z=cal_best(self.decoder,z,y_,9,18,beam_n)
        z=cal_best(self.decoder,z,y_,0,9,beam_n)
        z=cal_best(self.decoder,z,y_,18,27,beam_n)
        z=cal_best(self.decoder,z,y_,9,18,beam_n)
        z=cal_best(self.decoder,z,y_,0,9,beam_n)
        z=cal_best(self.decoder,z,y_,18,27,beam_n)
        z=cal_best(self.decoder,z,y_,9,18,beam_n)
        z=cal_best(self.decoder,z,y_,0,9,beam_n)
        z=cal_best(self.decoder,z,y_,0,9,1)
        out=z[:,-1]
        ones=torch.ones_like(out)[:,:2]
        zeros=torch.zeros_like(out)[:,:1]
        out=torch.cat([out,ones,zeros],-1)
        return out//0.50001
encoder4=Encoder()
class Encoder(torch.nn.Module):
    def __init__(self,d=512):
        super(Encoder, self).__init__()
        self.encoder0=copy.deepcopy(encoder0).cuda()
        self.encoder1=copy.deepcopy(encoder1).cuda()
        self.encoder2=copy.deepcopy(encoder2).cuda()
        self.encoder3=copy.deepcopy(encoder3).cuda()
        self.encoder4=copy.deepcopy(encoder4).cuda()
    def forward(self,x,n_iter=100):
        y_=x.reshape([x.shape[0],13,-1])
        out0=self.encoder0(x,n_iter)
        y0=self.encoder0.decoder(out0)
        score0=complex_corr(y0,y_).mean(-1)
        
        out1=self.encoder1(x)
        y1=self.encoder1.decoder(out1)
        score1=complex_corr(y1,y_).mean(-1)
        
        out2=self.encoder2(x)
        y2=self.encoder2.decoder(out2)
        score2=complex_corr(y2,y_).mean(-1)
        
        out3=self.encoder3(x)
        y3=self.encoder3.decoder(out3)
        score3=complex_corr(y3,y_).mean(-1)
        
        
        out4=self.encoder4(x)
        y4=self.encoder4.decoder(out4)
        score4=complex_corr(y4,y_).mean(-1)
        
        index=torch.stack([score0,score1,score2,score3,score4],-1).argmax(-1)
        
        out=torch.stack([out0,out1,out2,out3,out4],1)
        out=batch_gather(out,index)
        return out
        
        
class Decoder(torch.nn.Module):
    def __init__(self,d=512):
        super(Decoder, self).__init__()
        self.cnt=0
        self.if_confuse=False
    def forward(self,x):
        out=list()
        for i in x:
            if int(i[-1])==0:
                if int(i[-2])==0:
                    decoder=self.decoder0
                else:
                    if int(i[-3])==0:
                        decoder=self.decoder3
                    else:
                        decoder=self.decoder4
            else:
                if int(i[-2])==0:
                    decoder=self.decoder1
                else:
                    decoder=self.decoder2
            out.append(decoder(i.unsqueeze(0)))
        out=torch.cat(out,0)
        if self.if_confuse:
            ones=torch.ones_like(out)

            i=torch.arange(self.cnt,self.cnt+out.shape[0],device=out.device,dtype=out.dtype)%2
            out=torch.einsum('bwd,b->bwd',out,i)+torch.einsum('bwd,b->bwd',ones,1-i)
            self.cnt+=out.shape[0]
        return do_norm(out)
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.decoder.decoder0=self.encoder.encoder0.decoder
        self.decoder.decoder1=self.encoder.encoder1.decoder
        self.decoder.decoder2=self.encoder.encoder2.decoder
        self.decoder.decoder3=self.encoder.encoder3.decoder
        self.decoder.decoder4=self.encoder.encoder4.decoder
#         self.decoder.if_confuse=True
    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out