import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpmath import mp, mpf, loggamma, exp, log as mlog
import math
mp.dps=80
def lB(a,b): return loggamma(a)+loggamma(b)-loggamma(a+b)
def branch(T):
    n1=(3**T+5)//2; n2=3**T-1; E1=3**T+2; E12=2**(T+2)-4; E2=2*3**T+2-2**(T+2)
    k1=2*3**T+2**(T+2); k2=4*3**T-2**(T+2); return n1,n2,E1,E2,E12,k1,k2
def Nof(T): return 3*(3**T+1)//2
def logR_plain(T,a):
    n1,n2,E1,E2,E12,_,_=branch(T); a=mpf(a); Nn=n1+n2
    m1=n1*(n1-1)//2;m2=n2*(n2-1)//2;m12=n1*n2;M=Nn*(Nn-1)//2;E=E1+E2+E12
    return float((lB(E1+a,m1-E1+a)+lB(E2+a,m2-E2+a)+lB(E12+a,m12-E12+a)+lB(n1+a,n2+a)
                 -2*lB(a,a)-lB(E+a,M-E+a)-lB(a,Nn+a)).real)
def logR_dc(T,a):
    _,_,E1,E2,E12,k1,k2=branch(T); a=mpf(a); m=E1+E2+E12; tm=k1+k2
    O11=mpf(k1*k1)/(2*tm);O22=mpf(k2*k2)/(2*tm);O12=mpf(k1*k2)/tm;Ot=O11+O22+O12
    lZ=lambda e,O: loggamma(e+a)-(e+a)*mlog(O+a)
    return float((lZ(E1,O11)+lZ(E2,O22)+lZ(E12,O12)-lZ(m,Ot)+2*(a*mlog(a)-loggamma(a))).real)
def Ps(lr): return float(1/(1+exp(-mpf(lr))))

Ts=list(range(2,13)); Ns=[Nof(T) for T in Ts]
styles={0.5:':',1.0:'-',2.0:'--'}
plt.rcParams.update({'font.size':9,'font.family':'serif','mathtext.fontset':'cm',
                     'axes.linewidth':0.8,'xtick.direction':'in','ytick.direction':'in'})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.0,2.9))

for a in (0.5,1.0,2.0):
    ax1.semilogx(Ns,[Ps(logR_plain(T,a)) for T in Ts],'o'+styles[a],color='#c0392b',ms=3,lw=1.1,mfc='white')
    ax1.semilogx(Ns,[Ps(logR_dc(T,a)) for T in Ts],'s'+styles[a],color='#2471a3',ms=3,lw=1.1,mfc='white')
ax1.axhline(0.99,color='0.6',lw=0.7,ls='-.')
ax1.axvline(1095,color='#c0392b',lw=0.7,ls=(0,(4,2)),alpha=0.7)
ax1.axvline(123,color='#2471a3',lw=0.7,ls=(0,(4,2)),alpha=0.7)
ax1.set_xlabel(r'$n$'); ax1.set_ylabel(r'$P(\mathrm{split})$')
ax1.set_ylim(-0.03,1.05); ax1.text(0.05,0.9,'(a)',transform=ax1.transAxes,fontweight='bold')
ax1.text(1150,0.35,r'$r_\kappa^{\rm plain}$',color='#c0392b',fontsize=8)
ax1.text(60,0.62,r'$r_\kappa^{\rm dc}$',color='#2471a3',fontsize=8)
ax1.plot([],[],'-',color='#c0392b',label='plain SBM')
ax1.plot([],[],'-',color='#2471a3',label='degree-corrected')
ax1.legend(frameon=False,fontsize=7.5,loc='center right')

for a in (0.5,1.0,2.0):
    ax2.loglog(Ns,[abs(logR_plain(T,a)) for T in Ts],'o'+styles[a],color='#c0392b',ms=3,lw=1.1,mfc='white')
    ax2.loglog(Ns,[abs(logR_dc(T,a)) for T in Ts],'s'+styles[a],color='#2471a3',ms=3,lw=1.1,mfc='white')
xx=np.array([1e3,1e6])
ax2.loglog(xx,(math.log(3)-2/3*math.log(2))*xx,'-',color='#c0392b',lw=0.7,alpha=0.5)
ax2.loglog(xx,(2*math.log(3)-4/3*math.log(2))*xx,'-',color='#2471a3',lw=0.7,alpha=0.5)
ax2.set_xlabel(r'$n$'); ax2.set_ylabel(r'$|\log R|$')
ax2.text(0.05,0.9,'(b)',transform=ax2.transAxes,fontweight='bold')
ax2.text(4e4,4e3,r'$\sim(2\ln3-\frac{4}{3}\ln2)n$',color='#2471a3',fontsize=7,rotation=32)
ax2.text(6e4,3e2,r'$\sim(\ln3-\frac{2}{3}\ln2)n$',color='#c0392b',fontsize=7,rotation=32)
plt.tight_layout(pad=0.5)
plt.savefig('pseudofractal_figure.pdf',bbox_inches='tight')
plt.savefig('pseudofractal_figure.png',dpi=200,bbox_inches='tight')
print("figure saved")
