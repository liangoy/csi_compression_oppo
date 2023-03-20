from modelDesign import *

model_id='1'
task='1'
data_dir=f'../raw_data'
n_iter=313000 
'''
init
'''
model=AutoEncoder().encoder.__getattr__(f'encoder{model_id}').cuda()
saver=Saver(model,f'/tmp/encoder_{task}.pth.tar{model_id}')
optimizer=torch.optim.Adam(model.parameters(),lr=3e-3)

data=sio.loadmat(f'{data_dir}/W{task}.mat')["W"]#[1000,8,13]
data=data.transpose((0,2,1))
data=np.stack([data.real,data.imag],-1).reshape([-1,13,16])
index=sorted(range(len(data)),key=lambda x:md5(str(x).encode()).hexdigest())
train_data=torch.tensor(data[index],dtype=torch.float32).cuda()

n=nnv(train_data.reshape([-1,16]))
bg=cal_nn(n.unsqueeze(0).unsqueeze(0).repeat(1000,13,1),train_data[:])
bg=do_norm(bg)
model.encoder.background[:1000]=bg

'''
train
'''
class Dataset(torch.utils.data.Dataset):
    def __init__(self,H):
        super(Dataset, self).__init__()
        self.H=H
    def __getitem__(self,i):
        i=np.random.randint(1000)
        i2=np.random.randint(4)
        index=np.arange(0,52,4)+np.random.randint(0,4,13)
        n=self.H[i,:,:,index].conj().transpose(0,2,1)
        r=np.random.randn(4)+np.random.randn(4)*1j
        n=n*r
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
    optimizer.param_groups[0]['lr']=min(1e-3/10000*(step+1),1e-3)
    y_=do_norm(next(data_iter).cuda())
    
    y=model.decoder(model.encoder(y_))
    score=complex_corr(y,y_).mean()
    sim=complex_corr(y.unsqueeze(1),y).sort(1)[0][:,-2].max(-1)[0]
    sim=torch.clamp(sim,0.9,1)
    sim=sim.mean()
    loss=-score+sim*0.1
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
for step in range(100):
    if not model.training:
        model.train()
    optimizer.param_groups[0]['lr']=3e-5
    y_=do_norm(next(data_iter).cuda())
    
    y=model.decoder(model.encoder(y_))
    score=complex_corr(y,y_).mean()
    sim=complex_corr(y.unsqueeze(1),y).sort(1)[0][:,-2].max(-1)[0]
    sim=torch.clamp(sim,0.9,1)
    sim=sim.mean()
    loss=-score+sim*0.1
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(),0.01)
    optimizer.step()
    if step%10==0:
        state_dict_list.append(model.state_dict())
dic={key:sum([state_dict[key]for state_dict in state_dict_list])/len(state_dict_list) for key in state_dict_list[0].keys()}
model.load_state_dict(dic)
saver.save(f'../user_data/encoder_{task}.pth.tar{model_id}_')
