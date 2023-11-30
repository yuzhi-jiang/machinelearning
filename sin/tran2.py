from paddle import fluid as fl
import numpy as np
import matplotlib.pyplot as plt

def get_data(x):
    c,r = x.shape
    y = np.sin(x*3.14)+1+ (0.02*(2*np.random.rand(c,r)-1))
    return(y)

xs = np.arange(0,3,0.01).reshape(-1,1)
ys = get_data(xs)
xs = xs.astype('float32')
ys = ys.astype('float32')

"""plt.title("curve")
plt.plot(xs,ys)

plt.show()"""

x = fl.layers.Variable(name="x",shape=[1],dtype="float32")
y = fl.layers.Variable(name="y",shape=[1],dtype="float32")

l1 = fl.layers.fc(input=x,size=64,act="relu")
#l1 = fl.layers.fc(input=l1,size=16,act="relu")
pre = fl.layers.fc(input=l1,size=1)

loss = fl.layers.mean(
    fl.layers.square_error_cost(input=pre,label=y))

opt = fl.optimizer.Adam(0.1)
opt.minimize(loss)

exe = fl.Executor(
    fl.core.CPUPlace())
exe.run(fl.default_startup_program())

for i in range(1,4001):
    outs = exe.run(
        feed={x.name:xs,y.name:ys},
        fetch_list=[pre.name,loss.name])
    if(i%500==0):
        print(i," steps,loss is",outs[1])


plt.title("sin")
plt.plot(xs,ys)
plt.plot(xs,outs[0])
plt.show()
