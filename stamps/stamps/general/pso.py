import numpy as np
from six.moves import range


def pso(objf, xb, args=(), flag="m", pn=100, it=10, iw=0.3, sc=1.0, si=1.3,
        vmp=None):
    '''
    objf: callable func(x,*args)
          The objective function
    xb: value bound ex. ((10,20),(30,59),...,(12.4,33))
    args: tuple
          Extra arguments passed to objf, i.e. objf(x,*args)
    flag: "M" or "m", (M)aximum or (m)imimum of objf
    pn: int, particle number
    it: int, iteration times
    iw: float, inertia weight
    sc: float, self cognition
    si: float, social interaction
    vmp: max velocity proportion 
         ex. Vmax = boundrange * vmp
         
    return list of optvalue, ex. [15.0,35.0,...,27.5]
    '''
    
    lb,ub = list(map(np.array,zip(*xb)))
    xn=lb.size
    if vmp:
        pass
    else:
        vmp = 0.5 * 1./(pn**(1./xn))
    
    rdarray = np.random.rand(pn,xn)
    pos = rdarray * (ub-lb) + lb
    rdarray = 2 * np.random.rand(pn,xn) - 1
    vm=(ub - lb) * vmp #need vm later
    vel = rdarray * vm
    
    #calculate initial value
    #set self_optimal_fit and position
    slfoptfit = np.zeros(pn)
    slfoptpos = pos #init is the same
    for index_i,pos_i in enumerate(pos):
        slfoptfit[index_i] = objf(pos_i,*args)
        #slfoptpos[i] = pos_i
        
    #set social_optmal_fit and position
    socoptfit = slfoptfit[0]
    socoptpos = slfoptpos[0]
    if flag == "M":
        index = 0
        for pn_i in range(pn):
            if slfoptfit[pn_i] > socoptfit:
                socoptfit = slfoptfit[pn_i]
                index = pn_i
        socoptpos = slfoptpos[index]
    elif flag == "m":
        index = 0
        for pn_i in range(pn):
            if slfoptfit[pn_i] < socoptfit:
                socoptfit = slfoptfit[pn_i]
                index = pn_i
        socoptpos = slfoptpos[index]
        
    #do cycle
    determ = 0
    for it_i in range(it):
        vel = iw * vel + \
              sc * np.random.rand(pn,xn) * (slfoptpos - pos) + \
              si * np.random.rand(pn,xn) * (socoptpos - pos)
        for xn_i in range(vel[0].size):
            vel[:,xn_i] = np.where(abs(vel[:,xn_i]) > vm[xn_i],
                                      vel[:,xn_i]/abs(vel[:,xn_i]+10**-10)*vm[xn_i],
                                      vel[:,xn_i])
#        for vel_i in vel:
#            vel_i = np.where(abs(vel_i) > vm,vel_i/abs(vel_i)*vm,vel_i)
        pos += vel
        for xn_i in range(pos[0].size):
            pos[:,xn_i] = np.where(pos[:,xn_i] > ub[xn_i],
                                      ub[xn_i],
                                      pos[:,xn_i])
            pos[:,xn_i] = np.where(pos[:,xn_i] < lb[xn_i],
                                      lb[xn_i],
                                      pos[:,xn_i])
#        for pos_i in pos:
#            pos_i = np.where(pos_i > ub,ub,pos_i)
#            pos_i = np.where(pos_i < lb,lb,pos_i)
        slfoptfit_temp = np.zeros(pn)
#        slfoptpos_temp = pos
        for index_i,pos_i in enumerate(pos):
            slfoptfit_temp[index_i] = objf(pos_i,*args)
        if flag == "M":
            #compare self_optimal_fit
            if np.any(slfoptfit_temp > slfoptfit):
                optindex = np.where(slfoptfit_temp > slfoptfit)
                slfoptfit[optindex] = slfoptfit_temp[optindex]
                slfoptpos[optindex] = pos[optindex]
            #compare social_optimal_fit
            if np.any(slfoptfit > socoptfit):
                socoptfit = slfoptfit.max()
                socoptpos = pos[slfoptfit == socoptfit][0] 
                #need only one, the first one is choosen
        elif flag == "m":
#            if abs(slfoptfit.min()-socoptfit) <0.0001 and \
#               it_i > 20 and \
#               np.all(abs(pos[slfoptfit == slfoptfit.min()][0]-socoptpos) < 0.0001) :
#                determ += 1
#                if determ == 5:
#                    break
#            else:
#                determ = 0
            #compare self_optimal_fit
            if np.any(slfoptfit_temp < slfoptfit):
                optindex = np.where(slfoptfit_temp < slfoptfit)
                slfoptfit[optindex] = slfoptfit_temp[optindex]
                slfoptpos[optindex] = pos[optindex]
            #compare social_optimal_fit
            if np.any(slfoptfit < socoptfit):
                socoptfit = slfoptfit.min()
                socoptpos = pos[slfoptfit == socoptfit][0] 
                #need only one, the first one is choosen
    #print "iter_times = ",it_i,"objfuncvalue = ",socoptfit
    return socoptpos.tolist(), socoptfit

if __name__ == "__main__":
    def objf(x):
        y=x[1]
        x=x[0]
        return x**2+2*x*y+5+y**2+2*y+3 #min = 4
    xb = [[-90,30],[-2,30]]
    ans=pso(objf,xb,args=(),flag="m",
        pn=98,it=50,iw=0.3,sc=1.0,si=1.3,
        vmp=None)
    print (ans)
    
##    ans2=scipy.optimize.fminbound(objf,np.array([-90.,30.]),np.array([30.,60.]))
##    print ans2

