import torch
from torch.autograd import Variable
batch_n = 100#一个批次输入数据的数量
hidden_layer = 100
input_data = 1000#每个数据的特征为1000
output_data = 10

x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
y = Variable(torch.randn(batch_n,output_data),requires_grad=False)
#用Variable对Tensor数据类型变量进行封装的操作。requires_grad如果是False，表示该变量在进行自动梯度计算的过程中不会保留梯度值。
w1 = Variable(torch.randn(input_data,hidden_layer),requires_grad=True)
w2 = Variable(torch.randn(hidden_layer,output_data),requires_grad=True)

#学习率和迭代次数
epoch_n=50
lr=1e-6

for epoch in range(epoch_n):
    h1=x.mm(w1)#(100,1000)*(1000,100)-->100*100
    print(h1.shape)
    h1=h1.clamp(min=0)
    y_pred = h1.mm(w2)
    #y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred-y).pow(2).sum()
    print("epoch:{},loss:{:.4f}".format(epoch,loss.data))
    
#     grad_y_pred = 2*(y_pred-y)
#     grad_w2 = h1.t().mm(grad_y_pred)
    loss.backward()#后向传播
#     grad_h = grad_y_pred.clone()
#     grad_h = grad_h.mm(w2.t())
#     grad_h.clamp_(min=0)#将小于0的值全部赋值为0，相当于sigmoid
#     grad_w1 = x.t().mm(grad_h)
    w1.data -= lr*w1.grad.data
    w2.data -= lr*w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
    
#     w1 = w1 -lr*grad_w1
#     w2 = w2 -lr*grad_w2

