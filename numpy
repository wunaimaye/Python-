ndarray代码：

数组的创建和变换：
import numpy as np
'''
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(a.shape)
'''

'''
a = np.random.rand(3, 4)
print(a)
'''

'''
#arange生成的是整数类型
a = np.arange(10)
b = np.arange(1, 10, 2)
print(a)
print(b)
'''

'''
#若不用dtype定义类型，则ones，zeros和empty自动生成浮点型数组
a = np.ones((3, 4))
b = np.zeros((3, 4), dtype=np.int32)
c = np.empty((3, 4))
e = np.full((3, 4), 7)#生成一个3*4的全7数组
d = np.ones((2, 3, 4))#多维数组
print(a)
print(b)
print(c)
print(e)
print(d)
print(d.shape)
'''

'''
#eye(n)生成n*n矩阵，对角线为1，其余为0
a = np.eye(5)
print(a)
'''

'''
#b，c，d分别生成一个维数和a一样的，有各自元素的数组
a = np.array([[1, 2, 3], [5, 6, 7]])
print(a)
b = np.ones_like(a)
c = np.zeros_like(a)
d = np.full_like(a, 7)
print(b)
print(c)
print(d)
'''

'''
#若不用dtype定义类型，则linspace自动生成浮点型数组
a = np.linspace(1, 10, 4)#从1到10，等分为四个元素
b = np.linspace(1, 10, 4, endpoint=False)
print(a)
print(b)
c = np.concatenate((a, b))#连接两个数组
print(c)
'''

'''
#reshape()不改变原数组
a = np.ones((2, 3, 4), dtype=np.int32)
print(a)
print(a.reshape((3, 8)))#对a数组备份并返回一个(3，8)的数组
'''

'''
#resize()改变原数组
a = np.ones((2, 3, 4), dtype=np.int32)
a.resize((3, 8))
print(a)#返回改变后的数组a
'''

'''
#flatten(),ravel()不改变原数组
a = np.ones((2, 3, 4), dtype=np.int32)
print(a.flatten())#对a数组降维成压缩后的一维数组
print(a.ravel())#作用与flatten()相同，但返回数组的一个可视图
print(a)
'''

'''
#astype()可以将数组的元素类型进行改变
#astype()方法一定会创建一个新的数组(原始数据的一个拷贝)，即是两个数据类型一致
a = np.ones((2, 3, 4), dtype=np.int32)
print(a)
b = a.astype(np.float32)#将数组a的整型元素改变成浮点型
print(b)
'''

'''
#tolist()可以将数组转换成列表，且不改变原数组
a = np.full((2, 3, 4), 7, dtype=np.int32)
print(a.tolist())
print(a)
'''


数组的操作：
import numpy as np
'''
a = np.array([9, 8, 7, 6, 5])#元素下标从0开始
#索引
print(a[2])#索引第2个元素7
print(a[-2])#从后向前索引第2个元素
#切片
print(a[1:4])#从前往后索引不包含第4个元素，默认步长为1
print(a[-3:-1])#从后往前索引不包含第-1个元素，默认步长为1
print(a[0:4:2])#从前往后索引不包含第4个元素，步长为2
'''

'''
#多维数组的索引
a = np.arange(24).reshape((2, 3, 4))
print(a)
print(a[1, 2, 3])#索引第一维度中第1个元素中的第2个元素中的第3个元素
print(a[0, 1, 2])#索引第一维度中第0个元素中的第1个元素中的第2个元素
print(a[-1, -2, -3])#从后往前索引第一维度中第-1(即倒数第1)个元素中的第-2(即倒数第2)个元素中的第-3(即倒数第3)个元素
'''

'''
#多维数组的切片
a = np.arange(24).reshape((2, 3, 4))
print(a)
print(a[:, 1, -3])#“:”表示选取整一个为维度(即选取了例子中第一维度的0和1位置的元素)，然后选取这些元素的第1个元素的第-3(倒3)个元素组成新数组
print(a[:, 1:3, :])#第一维度全选，第二维度从第1切片到第2(不包括3)，第三维度全选
print(a[:, :, ::2])#第一维度全选,第二维度全选，第三维度全选但切取步长为2后的元素组成数组
'''


数组的运算：
import numpy as np
'''
#数组与标量之间的运算作用于数组的每一个元素
a = np.arange(24).reshape((2, 3, 4))
print(a)
print(a.mean())#输出a的标量
a = a / a.mean()
print(a)
'''

'''
#np.square(x), np.sqrt(x)不改变原数组
a = np.arange(24).reshape((2, 3, 4))
print(a)
b = np.square(a)
print(b)
c = np.sqrt(a)
print(c)
a = np.sqrt(a)
print(np.modf(a))#生成两个数组分别是a数组中小数部分和a数组中整数部分
'''

'''
#np.abs(x)，np.sign(), np.rint(x)，np.exp(x)不改变原数组
a = np.array([[1, 2, 3], [-4, -5, -6]])
print(np.abs(a))#取绝对值
print(np.sign(a))#标记每个元素符号，正为1，负为-1，其他为0
print(np.ceil(a))
print(np.floor(a))
print(a)
b = np.array([[1.1, 2.4, 3.5], [-4.7, -5.3, -6.9]])
print(np.rint(b))#返回数组四舍五入后的值
print(np.exp(b))#返回每个元素的指数值
print(b)
c = np.array([[1, 2, 3], [4, 5, 6]])
#此类函数也不改变原数组的元素值
print(np.log(c))#饭hi自然对数e
print(np.log10(c))#基于10的对数
print(np.log2(c))#基于2的对数
print(c)
'''

'''
a = np.arange(24).reshape((2, 3, 4))
b = np.sqrt(a)
print(a)
print(b)
print(np.maximum(a, b))#返回较大的数组
print(a > b)#返回两个数组各元素的比较，返回True或False
'''

数据存取处理：

数据的CSV文件存取（弊端是CSV只能有效存储一维和二维数组）：
(np.savetxt()和np.loadtxt()只能有效存取一维和二维数组)

import numpy as np
'''
#np.savetxt(frame, array, fmt="x", delimiter=None)存储数据
#frame：文件，字符串或生成器，也可以是.gz或.bz2的压缩文件(即有大量代码时使用)
#array：存入文件的数组
#fmt：写入文件的格式，例如：%d，%.2f，%.18e、
#delimiter：分割字符串，默认是任何空格
#a = np.arange(100).reshape(5, 20)
a = np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]])
np.savetxt("D:/a.CSV", a, fmt="%.1f", delimiter=",")
'''

'''
#np.loadtxt(frame, dtype=np.int32, delimiter=',', unpack=False)读入数据
#frame：文件，字符串或生成器，也可以是.gz或.bz2的压缩文件(即有大量代码时使用)
#dtype：以何种形式读入数据类型，可选
#delimiter：分割字符串，默认是任何空格
#unpack：如果是True，读入属性将分别写入不同的变量
b = np.loadtxt("D:/a.CSV", delimiter=',')
print(b)
c = np.loadtxt("D:/a.CSV", dtype=np.int32, delimiter=',')
print(c)
'''


多维数据的存取：
import numpy as np

'''
#np.tofile(frame, sep='', format='%d')存储多维数组
#frame：文件或字符串
#sep：数据分割字符串，如果是空串，写入文件为二进制
#format：写入数据的格式

#np.fromfile(frame, dtype=np.float, count=-1, sep='')读取多维数组，但维度会丢失，得在函数后用reshape()恢复维度
##frame：文件或字符串
#dtype：读取的数据类型
#count：读入的元素个数，-1表示读入整个文件
#sep：数据分割字符串，如果是空串，写入文件为二进制

b = np.arange(100).reshape(5, 10, 2)
b.tofile("D:/b.dat", sep=",", format="%d")
c = np.fromfile("D:/b.dat", dtype=np.int32, sep=",")
print(c)
d = np.fromfile("D:/b.dat", dtype=np.int32, sep=",").reshape(5, 10, 2)
print(d)
'''

'''
b = np.arange(100).reshape(5, 10, 2)
b.tofile("D:/b.dat", format="%d")#没有用sep则生成一个二进制文件，数据无法直观表示，但二进制文件会比文本文件占用更小的空间
c = np.fromfile("D:/b.dat", dtype=np.int32).reshape(5, 10, 2)
print(c)
#np.tofile()和np.fromfile()需要配合使用，且存数据时要知道数据的类型和维度，以便读取时还原类型和用reshape()还原维度
#可以通过元数据文件来存储额外信息
'''

'''
#numpy的便携文件存取（使用该存取方式必须基于numpy自定义的文件格式）：

#np.save(frame, array)或np.savez(frame, array)存储数据
#frame：文件名，以.npy为扩展名，压缩扩展名为.npz
#array：数组变量

#np.load(frame)读取数据且直接还原原维度
#frame：文件名，以.npy为扩展名，压缩扩展名为.npz

a = np.arange(100).reshape(5, 10, 2)
print(a)
np.save("D:/a.npy", a)
b = np.load("D:/a.npy")
print(b)
'''

随机数函数：

import numpy as np
'''
a = np.random.rand(3, 4, 5)#根据(3, 4, 5)来创建一个三维的均匀分布随机数数组
print(a)
'''

'''
b = np.random.randn(3, 4, 5)#根据(3, 4, 5)来创建一个三维的标准正态分布随机数数组
print(b)
'''

'''
c = np.random.randint(100, 200, (3, 4))#从100到200之间根据(3, 4)创建一个二维的随机数数组
print(c)
'''

'''
np.random.seed(10)
d = np.random.randint(100, 200, (3, 4))
print(d)
np.random.seed(10)#使用随机数种，可以使在重复使用时得到一个相同的随机数种
d = np.random.randint(100, 200, (3, 4))
print(d)
'''

'''
a = np.random.randint(100, 200, (3, 4))
print(a)
np.random.shuffle(a)#根据数组a的第1轴进行随机排列，此函数会使得原数组发生变化
print(a)
np.random.shuffle(a)
print(a)
'''

'''
a = np.random.randint(100, 200, (3, 4))
print(a)
print(np.random.permutation(a))#根据数组a的第1轴产生一个新的乱序数组，此函数不改变原数组
print(np.random.permutation(a))
print(a)
'''

'''
#choice(a, size, replace=False,p)从数组a中以p为概率抽取元素，形成size形状的新数组
a = np.random.randint(100, 200, (8,))
print(a)
print(np.random.choice(a,(3,2)))#返回从数组a中随机抽取的元素组成一个3*2的新数组(元素可能有重复)
print(np.random.choice(a,(3,2), replace=False))#用replace=False可以使随机抽取时，元素不被重复抽到
print(np.random.choice(a,(3,2), p=a/np.sum(a)))#可以通过修改概率值p来随机抽取数据(此处p值修改后使得元素值越大的元素的被抽取的概率越大)
'''

'''
u = np.random.uniform(0, 10, (3, 4))#产生均匀分布的数组(0是起始值，10是结束值，(3, 4)是数组的形状)
print(u)
n = np.random.normal(10, 5, (3, 4))#产生具有正态分布的数组(10是均值，5是标准差，(3, 4)是数组的形状)
print(n)
m = np.random.poisson(30, (3, 4))#产生泊松分布的数组(30是随机事件发生率，(3, 4)是数组的形状)
print(m)
'''

统计函数和梯度函数：

统计函数：
import numpy as np
'''
#axis：给定轴(整数或元组)，为0时表示第一维度，为1时表示第二维度，以此类推
a = np.arange(15).reshape(3, 5)
print(a)
print(np.sum(a))#返回数组所有元素的和
print(np.mean(a, axis=1))#axis=1表示求第二维度上元素的平均值，返回各元素的平均值(该维度的每个元素有内有五个元素)
print(np.mean(a, axis=0))#axis=0表示求第一维度上元素的平均值，返回各元素的平均值(该维度有三个元素，其中又包含五个元素)
print(np.average(a, axis=0, weights=[10, 5, 1]))#指定轴axis=0第一维度上的元素，weights确定了权值，返回加权平均数
print(np.std(a))#返回数组a的标准差
print(np.var(a))#返回数组a的方差
'''

'''
b = np.arange(15, 0, -1).reshape(3, 5)
print(b)
print(np.max(b))#返回数组的最大元素
print(np.argmax(b))#返回数组最大元素的下标(该下标是数组降成一维后的下标)
print(np.unravel_index(np.argmax(b), b.shape))#重塑成多维下标
print(np.ptp(b))#返回数组的最大值和最小值的差
print(np.median(b))#返回数组的中位数
'''


梯度函数：
import numpy as np
'''
a = np.random.randint(0, 20, (5))
print(a)
print(np.gradient(a))#返回每个元素的梯度值
b = np.random.randint(0, 50, (3, 5))
print(b)
print(np.gradient(b))#因为是二维数组，所以有多梯度，返回的第一个数组为最外层维度的梯度，返回的第二个数组为第二层维度的梯度
#如果一个数组是n维的，那么np.gradient(x)会生成n个数组，每个数组代表其中一个元素在第n维度的梯度值
'''
