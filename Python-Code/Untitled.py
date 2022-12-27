#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[21]:


import sys
import os
import shutil
import gc
import numpy as np
import scipy.integrate as it
import subprocess as sub
import matplotlib.pyplot as plt
import linecache as l
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from scipy import fftpack


# # Matplotlib Settings

# In[ ]:


import matplotlib
from matplotlib import rc

# Setting font size
fontsize = 30
rc('font', **{'family':'serif', 'serif':['Times'], 'size': fontsize})
rc('pdf', fonttype=42)
rc('text', usetex=True)
params = {'axes.labelsize': fontsize,'axes.titlesize':fontsize, 'legend.fontsize': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
matplotlib.rcParams.update(params)
pot = 1
markers = ['o', '^', 's', 'v','>','<','1','2','3','8']
colors = ['k', 'r', 'steelblue','mediumseagreen','b','g','c','m','y','w']


# In[2]:


pot = 1
prodrun=5000000
dt=1.0
A=1890
kb=1.987204e-3  #real units
kCal2Joule=4184
avaga=6.02214179e23
fs2s=1.0e-15
A2m =1.0e-10
K_real2SI=A2m*A2m*avaga*fs2s/(kCal2Joule)  #kapitza
K_real2SIcond=kCal2Joule/(avaga*fs2s*A2m)  #conductivity


# In[10]:


def SmoothGrad(x_left, T_graL, x_mid, Tw, x_right, T_graR,x_leftfull,x_midfull,x_rightfull):
    
    pL=np.poly1d(np.polyfit(x_left,T_graL,1))
    pmid=np.poly1d(np.polyfit(x_mid,Tw,1))
    pR=np.poly1d(np.polyfit(x_right,T_graR,1))

    #x=np.concatenate([x_left,x_mid,x_right],axis=0)
    #p=np.concatenate([pL(x_left),pmid(x_mid),pR(x_right)],axis=0)
    L=np.column_stack((x_left,pL(x_left)))
    Lfull=np.column_stack((x_leftfull,pL(x_leftfull)))
    M=np.column_stack((x_mid,pmid(x_mid)))
    Mfull=np.column_stack((x_midfull,pmid(x_midfull)))
    R=np.column_stack((x_right,pR(x_right)))
    Rfull=np.column_stack((x_rightfull,pR(x_rightfull)))
    return L,M,R,Lfull,Mfull,Rfull


# In[11]:


def cutXT2(x_left, T_graL, x_mid, Tw, x_right, T_graR):
    n=np.size(x_left)
    #print(x_left)
    x_left=np.delete(x_left, [0,1,2], axis=0)
    x_left=x_left[0:6]
    n=np.size(x_right)
    x_right=np.delete(x_right, [0,n-1,n-2,n-3], axis=0)
    x_right=x_right[0:6]
    n=np.size(x_mid)
    x_mid=np.delete(x_mid,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,n-51,n-50,n-49,n-48,n-47,n-46,n-45,n-44,n-43,n-42,n-41,n-40,n-39,n-38,n-37,n-36,n-35,n-34,n-33,n-32,n-31,n-30,n-29,n-28,n-27,n-26,n-25,n-24,n-23,n-22,n-21,n-20,n-19,n-18,n-17,n-16,n-15,n-14,n-13,n-12,n-11,n-10,n-9,n-8,n-7,n-6,n-5,n-4,n-3,n-2,n-1])
    n=np.size(T_graL)
    T_graL=np.delete(T_graL, [0,1,2], axis=0)
    T_graL=T_graL[0:6]
    n=np.size(T_graR)
    T_graR=np.delete(T_graR, [0,n-1,n-2,n-3], axis=0)
    T_graR=T_graR[0:6]    
    n=np.size(Tw)
    Tw=np.delete(Tw,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,n-51,n-50,n-49,n-48,n-47,n-46,n-45,n-44,n-43,n-42,n-41,n-40,n-39,n-38,n-37,n-36,n-35,n-34,n-33,n-32,n-31,n-30,n-29,n-28,n-27,n-26,n-25,n-24,n-23,n-22,n-21,n-20,n-19,n-18,n-17,n-16,n-15,n-14,n-13,n-12,n-11,n-10,n-9,n-8,n-7,n-6,n-5,n-4,n-3,n-2,n-1])
    #print(x_left, T_graL, x_mid, Tw, x_right, T_graR)
    return x_left, T_graL, x_mid, Tw, x_right, T_graR


# In[12]:


def calcThermal(x_left, T_graL, x_mid, Tw, x_right, T_graR,Q_flux):
    slope_TgraL, c_TgraL=np.polyfit(x_left,T_graL,1)
    slope_Tw, c_Tw=np.polyfit(x_mid,Tw,1)
    slope_TgraR, c_TgraR=np.polyfit(x_right,T_graR,1)
    
    k_graL=-(Q_flux/(slope_TgraL))    #W/mK
    k_W=-(Q_flux/(slope_Tw))
    k_graR=-(Q_flux/(slope_TgraR))

    x_intL=-wall
    x_intLw=-wall
    x_intR=wall
    x_intRw=wall

    Tint_lGra=slope_TgraL*x_intL+c_TgraL
    Tint_lWat=slope_Tw*x_intLw+c_Tw
    Tint_rWat=slope_Tw*x_intRw+c_Tw
    Tint_rGra=slope_TgraR*x_intR+c_TgraR
    delTL= Tint_lGra-Tint_lWat
    delTR=Tint_rWat-Tint_rGra
    G_L=((Tint_lGra-Tint_lWat))/Q_flux    #m2K/W
    G_R=((Tint_rWat-Tint_rGra))/Q_flux
    print("Temperature drop left  =",delTL,'K')
    print("Temperature drop right  =",delTR,'K')
    return k_graL,k_W,k_graR,G_L,G_R 


# In[19]:


diranl="/home/arjun/Documents/TUTORIALS_LAMMPS/ionic_liquuid/binary_mixtures/IL_NEMD_DT/IL_models/EMIM/emim-bf4-tfsi/charged_system/NEMD/Normal/"
conrange= [0,100,200,300,400]
trange= [350]
crange=[0.1,0.2,0.3,0.4,0.5,0.6]
iruns = [1]
kap=[]
denpeakL=[]
denpeakR=[]
denpeakLz=[]
denpeakRz=[]
Qnet=[]
cf=[]
bulkden=[]
kleft=[]
kright=[]
lk=[]
lkleft=[]
lkright=[]
KvsC=[]
KvsM=[]
    #print(Tfluid,Q)
for con in conrange:
    for c in crange:
        Q=[]
        Tfluid=[]
        Tleft=[]
        Tright=[]
        den =[]
        for irun in iruns:
            T=350
            print(con,T,c)
            heat=np.loadtxt(diranl+str(c)+'/bf_'+str(con)+'/heatflux.'+str(irun)+"."+str(T)+".dat")
            #heat=np.loadtxt(diranl+str(c)+'/bf_'+str(con)+'/heatfluxfull.'+str(irun)+"."+str(T)+".dat")
           # print(heat[4999][0],heat[3999][0])
            #print(heat[4999][1],heat[3999][1])
            #Heat1=np.absolute(heat[4999][0]-heat[3999][0])
            #Heat2=np.absolute(heat[4999][1]-heat[3999][1])
            Heat1=np.absolute(heat[0])
            Heat2=np.absolute(heat[1])          
            print(Heat1,Heat2)
            Q_fluxreal=-((Heat1+Heat2)*0.5)/(prodrun*dt*A)
            Q_flux=Q_fluxreal
            print(Q_flux)
            Q.append(Q_flux)
            
            datafluid=np.loadtxt(diranl+str(c)+'/bf_'+str(con)+'/temp_fluid.'+str(irun)+"."+str(T)+".profile",skiprows=4)
#datafluid=np.loadtxt(diranl+'tmp_water.profile',skiprows=4)
            tempfluid=datafluid[:,3]
            Tfluid.append(tempfluid)
           # print(Tfluid)
            dataleft=np.loadtxt(diranl+str(c)+'/bf_'+str(con)+'/temp_gnc_left.'+str(irun)+"."+str(T)+".profile",skiprows=4)
#dataleft=np.loadtxt(diranl+'tmp_gnc_left.profile',skiprows=4)
            templeft=dataleft[:,3]
            Tleft.append(templeft)
            dataright=np.loadtxt(diranl+str(c)+'/bf_'+str(con)+'/temp_gnc_right.'+str(irun)+"."+str(T)+".profile",skiprows=4)
#dataright=np.loadtxt(diranl+'tmp_gnc_right.profile',skiprows=4)
            tempright=dataright[:,3]
            wall = dataright[:,1][0]
            Tright.append(tempright)
        
            denfluid=np.loadtxt(diranl+str(c)+'/bf_'+str(con)+'/dens_fluid.'+str(irun)+"."+str(T)+".profile",skiprows=4)
#datafluid=np.loadtxt(diranl+'tmp_water.profile',skiprows=4)
            density=denfluid[:,3]
            den.append(density)   
#print(Tfluid)
        Qmean= sum(Q)/len(Q)
        print(Qmean)
        tmeanfluid= np.mean(Tfluid, axis = 0)
        tmeanleft=np.mean(Tleft, axis = 0)
        tmeanright=np.mean(Tright, axis = 0)
        denavg = np.mean(den, axis = 0)
    #print(tmeanfluid,tmeanleft,tmeanright)

        zfluid=datafluid[:,1]
        zleft=dataleft[:,1]
        zright=dataright[:,1]
        zden = denfluid[:,1]
        zcord=np.concatenate([zleft,zfluid,zright],axis=0)
        temp=np.concatenate([templeft,tempfluid,tempright],axis=0)
        result="~/Documents/TUTORIALS_LAMMPS/ionic_liquuid/binary_mixtures/IL_NEMD_DT/IL_models/EMIM/emim-bf4-tfsi/charged_system/NEMD/NEMD_RESULTS/"
        directory = result
        parent_dir = "bf="+str(con)+'/c='+str(c)
        path = os.path.join(directory , parent_dir)
        print(path)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        np.savetxt(path+'/tempGrad.'+"."+str(T)+".dat", np.column_stack((zcord,temp)))
        np.savetxt(path+'/tempGradleft.'+"."+str(T)+".dat", np.column_stack((zleft,templeft)))
        np.savetxt(path+'/tempGradfluid.'+"."+str(T)+".dat", np.column_stack((zfluid,tempfluid)))
        np.savetxt(path+'/tempGradright.'+"."+str(T)+".dat", np.column_stack((zright,tempright)))
        np.savetxt(path+'/dengrad.'+"."+str(T)+".dat", np.column_stack((zden,denavg)))    

###  Trimming edges
        zleft_trim,templeft,zfluid_trim,tempfluid,zright_trim,tempright=cutXT2(zleft,templeft,zfluid,tempfluid,zright,tempright)
        print(templeft,tempright)
###  Smoothening data
        zcord_trim=np.concatenate([zleft_trim,zfluid_trim,zright_trim],axis=0)
        temp_trim=np.concatenate([templeft,tempfluid,tempright],axis=0)
#zmidfull=np.linspace(fluid[0],fluid[1],num=fluid[2])
        left,mid,right,leftfull,midfull,rightfull=SmoothGrad(zleft_trim,templeft,zfluid_trim,tempfluid,zright_trim,tempright,zleft,zfluid,zright)
        np.savetxt(path+'/smoothGradleft.'+str(T)+"."+".dat", leftfull)
        np.savetxt(path+'/smoothGradright.'+str(T)+"."+".dat", rightfull)
        np.savetxt(path+'/smoothGradfluid.'+str(T)+"."+".dat", midfull)

###  Kapitza and conductivity
        C_left,C_fluid,C_right,K_left,K_right=calcThermal(zleft_trim,templeft,zfluid_trim,tempfluid,zright_trim,tempright,Q_flux)
        C_left=C_left*K_real2SIcond
        C_right=C_right*K_real2SIcond
        C_fluid=C_fluid*K_real2SIcond
        K_left=K_left*K_real2SI
        K_right=K_right*K_real2SI
        C_solid=0.5*(C_left+C_right)
        K=0.5*1000000000*(K_left+K_right)
        l_kleft = C_fluid *K_left*1000000000
        l_kright = C_fluid * K_right*1000000000
        l_k = 0.5*(l_kleft+l_kright)
        np.savetxt(path+'/thermalCond.'+"."+str(T)+".dat",np.vstack((C_left,C_fluid,C_right,C_fluid,C_solid)))  
        np.savetxt(path+'/boundaryCond.'+"."+str(T)+".dat", np.vstack((K_left,K_right,K))) 
        print("##########",T,pot,"#########################")
        #print("Conductivity left  =",C_left,'W/mK')
        #print("Conductivity right =",C_right,'W/mK')
        #print("Conductivity Fluid =",C_fluid,'W/mK')
        #print("Conductivity Solid =",C_solid,'W/mK')
        print("Kapitza left       =",K_left,'m2K/W')
        print("Kapitza length left       =",l_kleft,'nm')
        print("Kapitza right      =",K_right,'m2K/W')
        print("Kapitza length right      =",l_kright,'nm')
        print("heatflux    =",Qmean,'W/m^2')
        #print("Kapitza Average    =",K,'m2K/GW')
        #print("Kapitza length Average    =",l_k,'nm')
        kap.append(K)
        kleft.append(K_left)
        kright.append(K_right)
        lk.append(l_k)
        lkleft.append(l_kleft)
        lkright.append(l_kright)
        KvsC.append((con,c,K_left,K_right))
        KvsM.append((c,con,K_left,K_right))
        print("###################################")
        data = pd.read_csv(path+'/dengrad.'+"."+str(T)+".dat",sep='\s+',header=None)
        data = pd.DataFrame(data)
        x = data[0]
        y = data[1]
        bins= int(np.size(y)/5)
        dmaxL=max(y[:bins])
        dmaxR=max(y[-bins:])
        yleft=y[:bins]
        yright=y[-bins:]
        dmaxLz=x[yleft.argmax()]
        dmaxRz=x[bins-int(yright.argmax())]        
        print(dmaxL,dmaxLz,dmaxR,dmaxRz)
        Qnet.append(Qmean)
        denpeakL.append(dmaxL)
        denpeakR.append(dmaxR)
        denpeakLz.append(dmaxLz)
        denpeakRz.append(dmaxRz)
        cf.append(C_fluid)
        z=y[bins:]
        zz=z[:-bins]
        zzz=np.mean(zz)
        bulkden.append(zzz)
     #   plt_1 = plt.figure(figsize=(10, 10))
        figure, axis = plt.subplots(1, 2)
        figure.set_size_inches(15, 5)
        axis[0].scatter(zleft_trim, templeft,color='b',label = 'Cold side')
        axis[0].scatter(zfluid_trim, tempfluid,color='brown',label = 'Fluid')
        axis[0].scatter(zright_trim, tempright,color='r',label = 'Hot side')
        #axis[0].set_title("Temperature profile@"+str(T))
        axis[1].plot(x, y,'b')
        #axis[1].set_title("Density Gradient_"+str(dmaxL))
        axis[0].legend(loc = 'lower right', shadow = True, 
            handlelength = 1.2, fontsize = 'large', borderaxespad = 0.7,ncol=1,frameon=True)
        axis[0].set_xlabel("z $(A^0)$",fontweight = 'bold',fontsize=12)
        axis[1].set_xlabel("z $(A^0)$",fontweight = 'bold',fontsize=12)
        axis[1].set_ylim(0,10)
        axis[0].set_ylabel("Temperature $(k)$",fontweight = 'bold',fontsize=12)
        axis[1].set_ylabel("Mass density $(kg/m^3)$",fontweight = 'bold',fontsize=12)
        extent = axis[0].get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        plt.savefig(diranl+str(c)+'/bf_'+str(con)+'temp_pr'+str(c)+'.'+str(con)+'.png',dpi=600, bbox_inches=extent.expanded(1.25, 1.3))
        extent = axis[1].get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        plt.savefig(diranl+str(c)+'/bf_'+str(con)+'dens_pro'+str(c)+'.'+str(con)+'.png',dpi=600, bbox_inches=extent.expanded(1.25, 1.3))        
        plt.savefig(diranl+str(c)+'/bf_'+str(con)+"fig"+str(c)+'.'+str(con)+'.png')
        plt.show()
        plt.close()
#print(kap,kleft,kright,denpeakL,denpeakLz,denpeakR,denpeakRz,bulkden,Qnet)
np.savetxt('data_normal'+".dat", np.column_stack((kap,kleft,kright,denpeakL,denpeakLz,denpeakR,denpeakRz,bulkden,Qnet)),header='Kapitza k_left k_right peak_denL peak_denLz peak_denR peak_denRz bulk_den Qtotal')
np.savetxt('K_vs_charge.dat',KvsC,header='BF Charge k_left k_right')
np.savetxt('K_vs_molarity.dat',KvsM,header='Charge BF k_left k_right')


# ## 1. TBR vs Surface Charge of Carbon atoms

# In[64]:


fig, ax = plt.subplots()
conrange= [0,100,200,300,400]
for con in conrange:
    data = pd.read_csv("K_vs_charge.dat",sep='\s+',header=None)
    data = pd.DataFrame(data)
    select_con = data.loc[data.iloc[:,0] == con]
    x = select_con[1]
    y1 = select_con[2]
    y2 = select_con[3]    
    plt.title('Kapitza resistance Vs surface charge for Bf4 concentration='+str(con))
    plt.plot(x, y1,color="red", marker="o",  linestyle="--",label='kapitza_left')
    plt.plot(x, y2,color="blue", marker="o",  linestyle="--",label='kapitza_right')
    plt.xlabel('surface charge(C/m^2)', style='italic')
    plt.ylabel('Kapitza resistance(m^2K/W)', style='italic')
    plt.show()


# ## 2. TBR vs Mixture Ratio

# In[63]:


crange=[0.1,0.2,0.3,0.4,0.5,0.6]
for c in crange:
    data = pd.read_csv("K_vs_molarity.dat",sep='\s+',header=None)
    data = pd.DataFrame(data)
    select_con = data.loc[data.iloc[:,0] == c]
    x = select_con[1]
    y1 = select_con[2]
    y2 = select_con[3]    
    plt.title('Kapitza resistance Vs BF4 concentration for surface charge='+str(c))
    plt.plot(x, y1,color="red", marker="o",  linestyle="--",label='kapitza_left')
    plt.plot(x, y2,color="blue", marker="o",  linestyle="--",label='kapitza_right')
    plt.xlabel('surface charge(C/m^2)', style='italic')
    plt.ylabel('Kapitza resistance(m^2K/W)', style='italic')
    plt.show()


# ## 3. EDL Structure for different Surface Charge

# In[ ]:


import MDAnalysis
from MDAnalysis.analysis.density import density_from_Universe
from MDAnalysis.analysis.lineardensity import LinearDensity
import numpy as np
from itertools import count

class Smooth_Density(object):
    
    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)
            
    def density(self, psf, dcd, segid, resname):
        u = MDAnalysis.Universe(psf, dcd)
        mol = u.select_atoms(segid)

        ldens = LinearDensity(mol, grouping ='atoms', binsize=0.5)
        ldens.run()
        D = ldens.results
        x = np.linspace(1,np.size(D['x']['pos']),np.size(D['x']['pos']))
        y = np.linspace(1,np.size(D['y']['pos']),np.size(D['y']['pos']))
        z = np.linspace(1,np.size(D['z']['pos']),np.size(D['z']['pos']))
        return x, D['x']['pos'], y, D['y']['pos'], z, D['z']['pos']

    def density_smooth(self):
        
        folder="T"+str(self.T)+"P"+str(self.P)+"_dT"+str(self.dT)+"_"+str(self.index)
        psf_wat = self.wat_input_files_path+folder+'/inputFiles/waterGnc.psf'
        dcd_wat = self.wat_input_files_path+folder+'/outputFiles/waterGnc.dcd'
        
        x, Dx, y, Dy, z, Dz = self.density(psf_wat, dcd_wat, self.wat_segid, self.wat_resname)
        
#         #Saving the density data
#         fname = self.wat_output_files_path+'water_Density_T'+str(T)+'.dat'
#         np.savetxt(fname, np.column_stack((z, Dz)))
        
        axis = self.axis

        if (axis == "x"):
            wat_X = x/self.conv_factor 
            wat_Den = Dx/self.wat_norm
        elif (axis == "y"):
            wat_X = y/self.conv_factor 
            wat_Den = Dy/self.wat_norm
        else:
            wat_X = z/self.conv_factor 
            wat_Den = Dz/self.wat_norm
        
    def read_denstiy(self, T):
        fname = self.wat_output_files_path+'water_Density_T'+str(T)+'.dat'
        wat_data = np.loadtxt(fname)
        wat_X = wat_data[:,0]/self.conv_factor 
        wat_Den = wat_data[:,1]/np.mean(wat_data[20:60,1])
        
        return wat_X, wat_Den


# ## 4. EDL Structure for different Mixture Ratio

# In[ ]:




