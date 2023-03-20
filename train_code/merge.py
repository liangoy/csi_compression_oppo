from modelDesign import *

model = AutoEncoder()


for task in ['1','2']:

    model.encoder.encoder0.load_state_dict(torch.load(f'../user_data/encoder_{task}.pth.tar0_')['state_dict'])
    model.encoder.encoder1.load_state_dict(torch.load(f'../user_data/encoder_{task}.pth.tar1_')['state_dict'])
    model.encoder.encoder2.load_state_dict(torch.load(f'../user_data/encoder_{task}.pth.tar2_')['state_dict'])
    model.encoder.encoder3.load_state_dict(torch.load(f'../user_data/encoder_{task}.pth.tar3_')['state_dict'])

    torch.save({'state_dict':model.encoder.state_dict()},f'../user_data/encoder_{task}.pth.tar')
    torch.save({'state_dict':model.decoder.state_dict()},f'../user_data/decoder_{task}.pth.tar')
    
    print(f'task{task} has been merged')