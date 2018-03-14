#--coding:utf-8--
#!/usr/bin/python
# -*- coding: utf-8 -*-
#PESA2算法 by Lo Rain ,qq:771527850,E-mail:luyueliang423@163.com
import math
import numpy as np
import random
import numpy
import matplotlib.pyplot as plt
def Dominates(x, y):
    # b=all([x[0]<=y[0],x[1]<=y[1]])and any([x[0]<y[0],x[1]<y[1]])
    b = 0
    if x[0] <= y[0]:
        if x[1] <= y[1]:
            if x[2] <= y[2]:
                if x[0] < y[0]:
                    b = 1
                elif x[1] < y[1]:
                    b = 1
                elif x[2]<y[2]:
                    b = 1
    return b
def DetermineDomination(pop_Cost):  # for determine the dominatied pop in the orginal pop
    n = len(pop_Cost)
    pop_IsDominated1 = []
    for i in range(n):
        pop_IsDominated1.append(0)
    for i in range(n):
        for j in range(n):
            if j != i:
                if Dominates(pop_Cost[j], pop_Cost[i]):
                    pop_IsDominated1[i] = 1  # if the pop dominatied we write the tital "1"
                    break
    return pop_IsDominated1
def random_int_list(start, stop, length):  # return the matrix of random
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.uniform(start, stop))
        random_list.sort()
    return random_list
def CostFunction(x):  # fitness function for two objects
    t=1
    t2=1
    t3=1
    m=0
    n=0
    for i in range(len(x)):
        if i!=len(x)-1:
            t=t*x[i]
        if i !=len(x)-1 and i !=len(x)-1:
            t2=t2*x[i]
        if i !=len(x)-1 and i !=len(x)-2 and i !=len(x)-3:
            t3=t3*x[i]
        m=m+x[i]**2
        if i >1:
            n=(x[i]-0.5)**2-math.cos(20*math.pi*(x[i]-0.5))
    g=100*(len(x)-3+n)
    # f1=(0.5*t*(1+g))
    # f2=(0.5*t2*(1+g)*(1-x[len(x)-2]))
    # f3= (0.5*t3*(1+g)*(1-x[len(x)-3]))

    f1=0.5*x[0]*x[1]*(1+g)
    f2=0.5*x[0]*(1-x[1])*(1+g)
    f3=0.5*(1-x[0])*(1+g)
    #print f2,f3
    return [f1,f2,f3]
def move(pop, C,lb,ub):
    pop_j = pop
    for i in range(len(pop)):
        dt = (2 * round(random.random()) - 1) * random.random()
        pop_j[i] = pop[i] + C * dt
        pop_j[i]=max(pop_j[i],lb[i])
        pop_j[i]=min(pop_j[i],ub[i])
    bf = CostFunction(pop_j)
    bh = CostFunction(pop)
    if Dominates(bf, bh):
        pop = pop_j
        value = bf
    else:
        value = bh
    return pop, value
def Creategrid(pop_Cost, nGrid, InflationFactor):
    zmin = [min([t[0] for t in pop_Cost]),min([t[1] for t in pop_Cost]),min([t[2] for t in pop_Cost])]  # constract the min and max value of each objects
    zmax = [max([t[0] for t in pop_Cost]), max([t[1] for t in pop_Cost]),max([t[2] for t in pop_Cost])]
    dz = [zmax[0] - zmin[0], zmax[1] - zmin[1],zmax[2]-zmin[2]]  # constract the distrance of min and max value of each objects
    alpha = InflationFactor / 2.0
    zmin = [zmin[0] - alpha * dz[0], zmin[1] - alpha * dz[1],zmin[2] - alpha * dz[2]]
    zmax = [zmax[0] + alpha * dz[0], zmax[1] + alpha * dz[1],zmax[2] + alpha * dz[2]]

    nObj = len(zmin)
    C1, C2 ,C3= [], [],[]
    x = numpy.linspace(zmin[0], zmax[0], nGrid + 1)  # obtain the Grid of two objects
    for k in range(len(x)):
        C1.append(x[k])
    x = numpy.linspace(zmin[1], zmax[1], nGrid + 1)
    for k in range(len(x)):
        C2.append(x[k])
    x = numpy.linspace(zmin[2], zmax[2], nGrid + 1)
    for k in range(len(x)):
        C3.append(x[k])
    C = [C1, C2, C3]
    empty_grid_N = numpy.zeros([len(C1), len(C2),len(C3)])
    t = 0
    for i in range(len(C1)):
        for j in range(len(C2)):
            for k in range(len(C3)):
                empty_grid_N[i][j][k] = t
                t = t + 1
    pop_Grid = numpy.zeros(len(pop_Cost))
    grid = numpy.zeros(t)
    for i in range(len(pop_Cost)):
        t1 = numpy.zeros(len(C1))
        t2 = numpy.zeros(len(C2))
        t3 = numpy.zeros(len(C2))
        for k in range(len(C1)):
            if pop_Cost[i][0]<=C1[k]:
                t1[k]=1
        for k in range(len(C1)):
            if pop_Cost[i][1]<=C2[k]:
                t2[k]=1
        for k in range(len(C1)):
            if pop_Cost[i][2]<=C3[k]:
                t3[k]=1
        # t1 = pop_Cost[i][0] <= numpy.array[C1]  # we must add the numpy.array,if not,it will putout only one value
        # t2 = pop_Cost[i][1] <= numpy.array[C2]
        # t3=  pop_Cost[i][2] <= numpy.array[C3]
        m = 0
        n = 0
        o = 0
        for k in range(len(C1)):
            if t1[k]:
                break
            else:
                m = m + 1
        for j in range(len(C2)):
            if t2[j]:
                break
            else:
                n = n + 1
        for j in range(len(C3)):
            if t3[j]:
                break
            else:
                o = o + 1
        pop_Grid[i] = empty_grid_N[m - 1][n - 1][o-1]
        grid[int(empty_grid_N[m - 1][n - 1][o-1])] = grid[int(empty_grid_N[m - 1][n - 1][o-1])] + 1  # because empty_grid_N[m-1][n-1] is flourt, so we must add the int
    return [pop_Grid, grid]
    #############################################################
    #############################################################
def TruncatePopulation(archive_Position, archive_Cost, archive_Grid, nArchive, grid):
    while len(
            archive_Grid) - nArchive > 0:  # in the fellow,we will select the pop in archive,for contralling the number of pop in archive
        t_i = np.argmax(grid)  # the positon of the pop
        t_value = max(grid)  # the number of pop in archive
        select_value = int(t_value * random.random())  # creat a random number for delete the pop
        t = 0
        ar_Grid = []
        ar_Position = []
        ar_Cost = []
        for j in range(len(archive_Cost)):
            if archive_Grid[j] == t_i:
                if t != select_value:  ##in the archive which have the most pop,we will delect one pop randomly
                    ar_Cost.append(archive_Cost[j])
                    ar_Grid.append(archive_Grid[j])
                    ar_Position.append(archive_Position[j])
                    t = t + 1
                else:
                    grid[t_i] = grid[t_i] - 1
                    t = t + 1
            else:
                ar_Cost.append(archive_Cost[j])
                ar_Grid.append(archive_Grid[j])
                ar_Position.append(archive_Position[j])
        archive_Position = ar_Position  # obtain the new archive
        archive_Grid = ar_Grid
        archive_Cost = ar_Cost
    return archive_Cost, archive_Grid, archive_Position
total_archive_Cost=[]
total_archive_Position=[]
for test2 in range(100):
    print test2
    p=30#搜索范围的维度
    s=20#细菌的个数
    Nc=20#去化操作的次数
    Ns=5#去化操作中单项运动的最大次数
    C=0.005#反转选定方向以后，单个细菌前进的长度
    Nre=4#复制操作的步骤
    Ned=2#驱散操作数
    Sr=s/2#每一代复制数
    Ped=0.5#细菌驱散的概率
    LowerBound=numpy.zeros(p)#每一维的上界
    UpBound=numpy.ones(p)#每一维的下边界
    nArchive=200
    InflationFactor=0.1
    nGrid=10
    d_attract=0.05
    ommiga_attract=0.05
    h_repellant=0.05
    ommiga_repellant=0.05
    pop_Position=numpy.zeros([s,p])
    pop_Cost=[]
    #VarMin,VarMax=-4,4#测试函数中维度的最大最小值

    for i in range(s):    #initialise the pop
        for j in range(p):
            pop_Position[i][j]=LowerBound[j]+(UpBound[j]-LowerBound[j])*random.random()
    for i in range(s):
        pop_Cost.append(CostFunction(pop_Position[i]))
    ObjectNumber=len(pop_Cost[0])
    MaxIt=50
    archive_Position=[]
    archive_Cost=[]
    for c in range(MaxIt):
        print c
        J = pop_Cost
        Jbe = pop_Position
        for h in range(Ned):
            for k in range(Nre):
                for j in range(Nc):
                    for i in range(s):
                        if c==1:
                            Jbest=pop_Cost[i]
                        k0=numpy.zeros(p)
                        for ji in range(p):
                            for si in range(s):
                                k0[ji]=k0[ji]+(pop_Position[i][ji]-pop_Position[si][ji])
                        Jcc=(-d_attract*math.exp(-ommiga_attract*(sum(k0)))+h_repellant*math.exp(-ommiga_repellant*sum(k0)))
                        J=pop_Cost
                        Jbe=pop_Position

                        J[i][0]=pop_Cost[i][0]+Jcc
                        J[i][1]=pop_Cost[i][1]+float(Jcc)
                        J[i][2] = pop_Cost[i][2] + float(Jcc)
                        if Dominates(J[j],pop_Cost[i]):
                            for li in range(p):
                                pop_Position[i][li]=pop_Position[i][li]+random.random()*C*(k0[li]/(s-1))
                                pop_Position[i][li] = max(pop_Position[i][li], LowerBound[li])
                                pop_Position[i][li] = min(pop_Position[i][li], UpBound[li])
                        m=0
                        pop_Position[i],pop_Cost[i]=move(pop_Position[i],C,LowerBound,UpBound)
                        while (m<Ns):
                            m=m+1
                            if Dominates(pop_Cost[i],J[i]):
                                J[i]=pop_Cost[i]
                                Jbe[i]=pop_Position[i]
                                pop_Position[i], pop_Cost[i] = move(pop_Position[i], C,LowerBound,UpBound)
                            else:
                                m=Ns

                    ndpop_Position = []  # clear the transfer matrix
                    ndpop_Cost = []
                    pop_IsDominated = DetermineDomination(J)
                    for i in range(len(J)):
                        if pop_IsDominated[i] == 0:
                            ndpop_Position.append(Jbe[i])
                            ndpop_Cost.append(J[i])
                    for ko in range(len(ndpop_Cost)):
                        archive_Position.append(ndpop_Position[ko])
                        archive_Cost.append(ndpop_Cost[ko])
                        # delete the dominated pop in the archive
                    ndpop_Position = []
                    ndpop_Cost = []
                    pop_IsDominated = DetermineDomination(archive_Cost)
                    for i in range(len(archive_Cost)):
                        if pop_IsDominated[i] == 0:
                            ndpop_Position.append(archive_Position[i])
                            ndpop_Cost.append(archive_Cost[i])
                    archive_Cost = []
                    archive_Position = []
                    for ko in range(len(ndpop_Cost)):
                        mt = 0
                        for kj in range(len(archive_Cost)):
                             # print len(archive_Cost)
                            if ko != 1:
                                if archive_Cost[kj][0] == ndpop_Cost[ko][0]:
                                    if archive_Cost[kj][1] == ndpop_Cost[ko][1]:
                                        if archive_Cost[kj][2] == ndpop_Cost[ko][2]:
                                            mt = 1
                                            break
                        if mt == 0:
                            archive_Position.append(ndpop_Position[ko])
                            archive_Cost.append(ndpop_Cost[ko])
                    [archive_Grid, grid] = Creategrid(archive_Cost, nGrid, InflationFactor)
                    if len(archive_Cost) > nArchive:
                        archive_Cost, archive_Grid, archive_Position = TruncatePopulation(archive_Position,archive_Cost,archive_Grid,nArchive,grid)
                pop_IsDominated = DetermineDomination(pop_Cost)
                for do in range(len(pop_Cost)):
                    for tt in range(len(pop_Cost)):
                        if pop_IsDominated[tt] == 0:
                            t_Position=pop_Position[tt]
                            break
                    if pop_IsDominated[do]==1:
                        if random.random()<0.5:
                            pop_Position[do]=t_Position
                    else:
                        t_Position=pop_Position[do]
                pop_Cost=[]
                for i in range(s):
                    pop_Cost.append(CostFunction(pop_Position[i]))
            for pe in range(len(pop_Position)):
                if Ped>random.random():
                    for j in range(p):
                        pop_Position[pe][j] = LowerBound[j] + (UpBound[j] - LowerBound[j]) * random.random()


        ndpop_Position = []  # clear the transfer matrix
        ndpop_Cost = []
        pop_IsDominated = DetermineDomination(J)
        for i in range(len(J)):
            if pop_IsDominated[i] == 0:
                ndpop_Position.append(Jbe[i])
                ndpop_Cost.append(J[i])
        for k in range(len(ndpop_Cost)):
            archive_Position.append(ndpop_Position[k])
            archive_Cost.append(ndpop_Cost[k])
            # delete the dominated pop in the archive
        ndpop_Position = []
        ndpop_Cost = []
        pop_IsDominated = DetermineDomination(archive_Cost)
        for i in range(len(archive_Cost)):
            if pop_IsDominated[i] == 0:
                ndpop_Position.append(archive_Position[i])
                ndpop_Cost.append(archive_Cost[i])
        archive_Cost = []
        archive_Position = []
        for k in range(len(ndpop_Cost)):
            mt = 0
            for j in range(len(archive_Cost)):
                 # print len(archive_Cost)
                if k != 1:
                    if archive_Cost[j][0] == ndpop_Cost[k][0]:
                        if archive_Cost[j][1] == ndpop_Cost[k][1]:
                            if archive_Cost[j][2]==ndpop_Cost[k][2]:
                                mt = 1
                                break
            if mt == 0:
                archive_Position.append(ndpop_Position[k])
                archive_Cost.append(ndpop_Cost[k])
        [archive_Grid, grid] = Creategrid(archive_Cost, nGrid, InflationFactor)
        if len(archive_Cost) > nArchive:
            archive_Cost, archive_Grid, archive_Position = TruncatePopulation(archive_Position, archive_Cost, archive_Grid,nArchive, grid)

    # from mpl_toolkits.mplot3d import Axes3D
    # x = [t[0] for t in archive_Cost]
    # y = [t[1] for t in archive_Cost]
    # z = [t[2] for t in archive_Cost]
    # fig=plt.figure(11)
    # ax = Axes3D(fig)
    # ax.scatter(x,y , z, c='r')  # 绘点
    # #plt.plot(x, y, 'ro')
    # plt.xlabel('f_1')
    # plt.ylabel('f_2')
    # plt.show()
    total_archive_Cost.append(archive_Cost)
    total_archive_Position.append(archive_Position)

mydata = []
mydata = total_archive_Cost

thefile = open("DTLZ1_bfo_basic_COST.txt", "w+")
for item in mydata:
    thefile.write("%s\n" % item)
thefile.close()
mydata = []
mydata = total_archive_Position
thefile = open("DTLZ1_bfo_basic_Position.txt", "w+")
for item in mydata:
    thefile.write("%s\n" % item)
thefile.close()
# x=[t[0] for t in archive_Cost]
# y=[t[1] for t in archive_Cost]
# print len(archive_Cost)
# #print y
#
# plt.figure(11)
# plt.plot(x,y, 'ro')
# plt.xlabel('f_1')
# plt.ylabel('f_2')
# plt.show()
#
















