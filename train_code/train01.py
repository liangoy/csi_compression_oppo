from modelDesign import *

model_id='0'
task='1'
data_dir=f'../raw_data'
n_iter=23000
'''
init
'''
model=AutoEncoder().encoder.__getattr__(f'encoder{model_id}').cuda()#读取模型
saver=Saver(model,f'/tmp/encoder_{task}.pth.tar{model_id}')
optimizer=torch.optim.Adam(model.parameters(),lr=3e-3)

data=sio.loadmat(f'{data_dir}/W{task}.mat')["W"]#[1000,8,13]
data=data.transpose((0,2,1))
data=np.stack([data.real,data.imag],-1).reshape([-1,13,16])
index=sorted(range(len(data)),key=lambda x:md5(str(x).encode()).hexdigest())
train_data=torch.tensor(data[index],dtype=torch.float32).cuda()#训练数据

n=nnv(train_data.reshape([-1,16]))#求train_data最大特征向量
v=nnv(train_data)#对各个样例求最大特征向量
v=do_norm(cal_nn(n.unsqueeze(0).repeat(1000,1),v))#对v进行旋转，使得v、n之间的欧式距离最小
v=v.unsqueeze(1).repeat(1,13,1)
bg=do_norm(cal_nn(v,train_data))#对train_data进行旋转，使得train_data、v之间的欧式距离最小，旋转后的train_data作为“背景”
model.encoder.background[:1000]=bg#用bg对模型的“背景”进行初始化

'''
train
'''
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
    
dataH=sio.loadmat(f'{data_dir}/H{task}.mat')["H"]
index=sorted(range(len(dataH)),key=lambda x:md5(str(x).encode()).hexdigest())
dataH=dataH[index]

dataset=Dataset(dataH)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=900,
    shuffle=False,
    num_workers=8,
    prefetch_factor=128,
    pin_memory=True
)
data_iter=iter(dataloader)


for step in range(n_iter):
    if not model.training:
        model.train()
    optimizer.param_groups[0]['lr']=min(3e-4/2000*(step+1),3e-4)
    y_=do_norm(next(data_iter).cuda())
#     r=do_norm(torch.rand_like(y_))
#     y_=do_norm(y_*50+r)#数据增强，随机噪声扰动
    
    y=model.decoder(model.encoder(y_))
    score=complex_corr(y,y_).mean()#余弦相识度
    sim=complex_corr(y.unsqueeze(1),y).max(-1)[0]
    ones=1-torch.diag(torch.ones(y.shape[0],dtype=y.dtype,device=y.device))
    sim=sim*ones
    sim=torch.clamp(sim,0.5,1).max(-1)[0].mean()#样本间的余弦显示度
    loss=-score+sim*0.5#最后的损失为最大化原数据与还原数据的余弦相似度以及最小化还原数据间的余弦相似度
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(),0.01)
    optimizer.step()
    if step%1000==0:
        print(__file__,step,float(score))

'''
save
'''
saver.save(f'../user_data/encoder_{task}.pth.tar{model_id}')


state_dict_list=list()
for step in range(100):#减小梯度做模型叠加
    if not model.training:
        model.train()
    optimizer.param_groups[0]['lr']=3e-5
    y_=do_norm(next(data_iter).cuda())
#     r=do_norm(torch.rand_like(y_))
#     y_=do_norm(y_*50+r)
    
    y=model.decoder(model.encoder(y_))
    score=complex_corr(y,y_).mean()
    sim=complex_corr(y.unsqueeze(1),y).max(-1)[0]
    ones=1-torch.diag(torch.ones(y.shape[0],dtype=y.dtype,device=y.device))
    sim=sim*ones
    sim=torch.clamp(sim,0.5,1).max(-1)[0].mean()
    loss=-score+sim*0.5
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(),0.01)
    optimizer.step()
    if step%10==0:
        state_dict_list.append(model.state_dict())
dic={key:sum([state_dict[key]for state_dict in state_dict_list])/len(state_dict_list) for key in state_dict_list[0].keys()}
model.load_state_dict(dic)
saver.save(f'../user_data/encoder_{task}.pth.tar{model_id}_')