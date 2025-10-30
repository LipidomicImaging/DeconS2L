from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet
sc = StandardScaler(with_mean=True)  
#不使用pre
thresh=2.27
limit=0.001
import seaborn as sns 
from sklearn.metrics import r2_score, get_scorer 
from sklearn.linear_model import  Lasso, Ridge, LassoCV,LinearRegression ,ElasticNetCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import  KFold, RepeatedKFold, GridSearchCV
import pandas as pd
import numpy as np
import winreg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso###导入lasso回归算法
from sklearn.metrics import r2_score
def get_sumarray(Forepre,Forefrag,index2,index):
    
    #[39,40,49,52,65,67,68,71,87,90,91]
    A=(Forepre[:, higher:][:,index2])
    B=Forefrag[:,index]
    #print(B.shape)
    sum_array_A =[]
    sum_array_B =[]
    for i in range(int(fore.shape[0]*0.3)):
        how=(np.random.choice(np.arange(0,fore.shape[0],1),9)).astype(np.int)
        sum_array_A.append(np.sum(A[how,:],axis=0)/9 )
        sum_array_B.append(np.sum(B[how],axis=0)/9)
    return np.array(sum_array_A),np.array(sum_array_B)
def lasso_regression(data,test,predictor,pre_y,alpha):
    lassoreg=Lasso(alpha=alpha,normalize=True,max_iter=1e5,fit_intercept=False)
    lassoreg.fit(data[predictors],data[pre_y])
    y_pred = lassoreg.predict(test[predictors])
    ret = [alpha]
    ret.append(r2_score(test[pre_y],y_pred)) #R方
    ret.append(mean_absolute_error(test[pre_y],y_pred)) #平均绝对值误差
    ret.extend(lassoreg.coef_) #模型系数
    return ret,y_pred
def norma(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
def norma1(x1):
    return (x1)/np.sum(x1,axis=0)+0.00000001,np.sum(x1,axis=0)#((np.max(x1,axis=0)-np.min(x1,axis=0))+0.001),
bin0=np.where(bin0<0,0,bin0)
#lassoreg=Lasso(alpha=0.01,max_iter=19000,tol=0.00000000000000001,positive=True)
lasso_alphas = np.linspace(0, 20, 2200)
def get_lambda(a,b):
    '''lasso = Lasso() 
    grid = dict() 
    grid['alpha'] = lasso_alphas 
    gscv = GridSearchCV( lasso, grid, scoring='neg_mean_absolute_error',cv=5, n_jobs=-1) 
    results = gscv.fit(sc.fit_transform(a), b)
#print('MAE: %.5f' % results.best_score_) 
#print('Config: %s' % results.best_params_)'''
    regr = ElasticNetCV(cv=10, positive=True,random_state=0)
    regr.fit(norma1(a)[0], b)
    return  regr.alpha_,regr.l1_ratio_#results.best_params_['alpha']
def get_sumarray(Forepre,Forefrag,index2,index):
    
    #[39,40,49,52,65,67,68,71,87,90,91]
    A=(Forepre[:, higher:][:,index2])
    B=Forefrag[:,index]
    
    return A[HOW,:],B[HOW]#,B#[HOW,:],B[HOW]
    #print(B.shape)
    '''sum_array_A =[]
    sum_array_B =[]
    dex=np.arange(0,Forepre.shape[0],1)
    
    num=1
    #int(Forepre.shape[0]/100)
    
    ratio=int(Forepre.shape[0]/num)
    print("ratio",ratio,int(ratio/10))
    for i in range(int(ratio/1)):
       # print(dex.shape,i)
        #print(dex.shape)
        how=(np.random.choice(np.arange(0,dex.shape[0],1),num,replace=False))
        #print(how)
        sum_array_A.append(np.sum(A[dex[how],:],axis=0)/num )
        sum_array_B.append(np.sum(B[dex[how]],axis=0)/num)
        dex=np.delete(dex,how)
    return np.array(sum_array_A),np.array(sum_array_B)'''
    #return A[:ratio*(1),:],B[:ratio*(1)]
def norma(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
def norma1(x1):
    return (x1)/(np.sum(x1,axis=0)+0.00001),np.sum(x1,axis=0)#((np.max(x1,axis=0)-np.min(x1,axis=0))+0.001),
bin0=bin0copy.copy()
FF=np.where(np.load("y_predict_2.npy")[:H*W]==0)[0]
y_predict=np.load("y_predict_5.npy")
#y_predict=np.squeeze(pd.read_excel("index.xlsx").values)[:H*(W-1)]
for J in range(0,5):
     #MYnet=Net().cuda()
    print("j",J)
    fore=np.where(y_predict[:H*W]==J)[0]
    
    print(fore.shape)
    precur=bin0[:,:]

    Fragement=bin0[:,:]
    p0=np.zeros((mz.shape[0],mz.shape[0]))
    Forepre=np.squeeze(precur[FF[fore],:])#
    Forefrag=np.squeeze(Fragement[FF[fore],:])#
    num=int(fore.shape[0]*1.0)
    if num>5000:
        num=5000
    print("num:",num)
    HOW=(np.random.choice(np.arange(0,fore.shape[0],1),num,replace=False))
    #HOW=[i*5 for i in range(1000)]
    #print(HOW)
    #Forefrag,coe=norma1(Fragement[FF[fore],:])
    cal=0
    pp=[]
    higher=np.where(mz>730)[0][0]
    higher_low=(mz.shape[0])
    low=(mz.shape[0])
    left=np.where(mz>141)[0][0]
    right=np.where(mz>210)[0][0]
    for i in range(left,right):
 #   [39,40,49,52,65,67,68,71,87,90,91]:
        fra=i
        
        print(mz[i])
    #[39,40,49,52,65,67,68,71,87,90,91]
        p1=np.zeros((mz.shape[0],mz.shape[0]))
        if np.sum(bin0[fore,i])<limit:
            continue
       # print(i)
        # if np.sum(Forefrag[:,i])==0:
         #   print(i)
         #   continue
        pp=[]
       
       
            
        #
        index=np.sort(np.argsort(np.sum(Forepre[:,higher:],axis=0))[-500:])

        print(mz[higher],mz[higher:][index])
       
        
       
        #p1[higher:,fra][index]=nnls(a,b)[0]
        #print(nnls(a,b)[1])
        for m in range(1): 

            a,b=get_sumarray((Forepre),(Forefrag),index,i)
            #if i==0:
            alpha,thresh=get_lambda(a,b)
            reg = ElasticNet(alpha=alpha, l1_ratio=thresh, positive=True,fit_intercept=False)
           
            reg.fit(norma1(a)[0],b)
          
            tmp=reg.coef_#*coe[i]*100000
            
            print(mz[i],np.sort(tmp)[-10:],mz[higher:][index][np.where(tmp>0)[-1000:]])
            tmp[np.argsort(tmp)[:-170]]=0
            p1[higher:,fra][index]=tmp
            pp.append(p1[:,fra])
        p0[:,fra]=np.sum(pp,axis=0)
       # print(p0[:,i])

    higher=np.where(mz>750)[0][0]
    higher_low=(mz.shape[0])
    low=(mz.shape[0])
    left=np.where(mz>500)[0][0]
    right=np.where(mz>660)[0][0]
    for i in range(left,right):
 #   [39,40,49,52,65,67,68,71,87,90,91]:
        fra=i
        
        print(mz[i])
    #[39,40,49,52,65,67,68,71,87,90,91]
        p1=np.zeros((mz.shape[0],mz.shape[0]))
        if np.sum(bin0[fore,i])<limit:
            continue
       # print(i)
        # if np.sum(Forefrag[:,i])==0:
         #   print(i)
         #   continue
        pp=[]
     
        if i>=left:
            higher=max(np.where(mz>(mz[i]+110))[0][0],higher)
            print(mz[higher],mz[i])
            if low<(mz.shape[0]):
                low=mz.shape[0]#min(np.where(mz>(mz[i]+230))[0][0],higher_low)
        else:
            higher=np.where(mz>750)[0][0]
            
        #
        index=np.sort(np.argsort(np.sum(Forepre[:,higher:low],axis=0))[-500:])

       
        for m in range(1):
            a,b=get_sumarray((Forepre),(Forefrag),index,i)
            #if i==192:
            alpha,thresh=get_lambda(a,b)
            lassoreg=Lasso(alpha=thresh,fit_intercept=False,max_iter=20000,tol=0.000000000000000000001,positive=True)
            reg = ElasticNet(alpha=alpha, l1_ratio=thresh,positive=True, fit_intercept=False)
           
            reg.fit(norma1(a)[0],b)#a*100000
          
            tmp=reg.coef_#*coe[i]*100000#*coe[i]
            #tmp=nnls(a,b)[0]*coe[i]
            print(mz[i],np.sort(tmp)[-10:],mz[higher:][index][np.where(tmp>0)[-150:]])
            tmp[np.argsort(tmp)[:-170]]=0
            p1[higher:low,fra][index]=tmp
          
            pp.append(p1[:,fra])
        p0[:,fra]=np.sum(pp,axis=0)
    #limit=0.1
    higher=np.where(mz>750)[0][0]
    low=(mz.shape[0])
    higher_low=(mz.shape[0])
    left=np.where(mz>660)[0][0]
    right=np.where(mz>750)[0][0]
    for i in range(left,right):#287
 #   [39,40,49,52,65,67,68,71,87,90,91]:
        fra=i
        #if i in pre:
           # continue
        print(mz[i],fore.shape)
    #[39,40,49,52,65,67,68,71,87,90,91]
        p1=np.zeros((mz.shape[0],mz.shape[0]))
        if np.sum(bin0[fore,i])<limit:
            continue
      
        pp=[]
     
        if i>=left:
            higher=max(np.where(mz>(mz[i]+40))[0][0],higher)
            if low<(mz.shape[0]):
                low=mz.shape[0]#min(np.where(mz>(mz[i]+350))[0][0],higher_low)
        else:
            higher=np.where(mz>750)[0][0]
            
        #
        index=np.sort(np.argsort(np.sum(Forepre[:,higher:low],axis=0))[-500:])#-int(0.4*(len(mz)-higher))
    
        for m in range(1): 
            a,b=get_sumarray((Forepre),(Forefrag),index,i)
           # if i==267:
            alpha,thresh=get_lambda(a,b)
            lassoreg=Lasso(alpha=thresh,fit_intercept=False,max_iter=20000,tol=0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000001,positive=True)
            reg = ElasticNet(alpha=alpha, l1_ratio=thresh,positive=True, fit_intercept=False)
          
            reg.fit(norma1(a)[0],b)#-noisea*100000
          
            tmp=reg.coef_#*coe[i]*100000#*coe[i]
            #tmp=nnls(a,b)[0]*coe[i]
            print(mz[i],np.sort(tmp)[-10:],mz[higher:][index][np.where(tmp>0)[-150:]])
            tmp[np.argsort(tmp)[:-150]]=0
            p1[higher:low,fra][index]=tmp
           # print(nnls(a,b)[1])
         
            pp.append(p1[:,fra])
        p0[:,fra]=np.sum(pp,axis=0)
    higher=np.where(mz>750)[0][0]
    higher_low=(mz.shape[0])
    left=np.where(mz>750)[0][0]
    right=np.where(mz>850)[0][0]
    for i in range(left,right):#287
 #   [39,40,49,52,65,67,68,71,87,90,91]:
        fra=i
        #if i in pre:
         #   continue
        print(mz[i],fore.shape)
    #[39,40,49,52,65,67,68,71,87,90,91]
        p1=np.zeros((mz.shape[0],mz.shape[0]))
        if np.sum(bin0[fore,i])<limit:
            continue
       # print(i)
        # if np.sum(Forefrag[:,i])==0:
         #   print(i)
         #   continue
        pp=[]
     
        if i>=left:
            higher=max(np.where(mz>(mz[i]+30))[0][0],higher)
            low=np.where(mz>899)[0][0]#min(np.where(mz>(mz[i]+90))[0][0],higher_low)
        else:
            higher=np.where(mz>750)[0][0]
            
        #
        index=np.sort(np.argsort(np.sum(bin0[fore,higher:low],axis=0))[-int(0.7*(len(mz)-higher)):])#-100-int(0.5*(len(mz)-higher))
      
        for m in range(1):
            a,b=get_sumarray((Forepre),(Forefrag),index,i)
            #if i==312:
            alpha,thresh=get_lambda(a,b)
            lassoreg=Lasso(alpha=thresh,fit_intercept=False,max_iter=20000,tol=0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000001,positive=True)
            reg = ElasticNet(alpha=alpha, l1_ratio=thresh, positive=True,fit_intercept=False)
         
            reg.fit(norma1(a)[0],b)#-noisea*100000
          
            tmp=reg.coef_#*coe[i]*100000#*coe[i]
          
            print(mz[i],np.sort(tmp)[-10:],mz[higher:][index][np.where(tmp>0)[-150:]])
            tmp[np.argsort(tmp)[:-150]]=0
            p1[higher:low,fra][index]=tmp
          
            pp.append(p1[:,fra])
        p0[:,fra]=np.sum(pp,axis=0)
       # print(p0[:,i])
       # print(p0[:,i])
       # print(p0[:,i])

       # print(p0[:,i])
    
    print(i)
    print("overall/p0__class%d.npy"%(J))
    np.save("data_1208/class5_all_1208/p0__class%d.npy"%(J),p0)
