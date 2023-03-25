import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


N=81
dt=0.001
tol_p=1e2
tol_v=1e-10
Re=100
dx=1.0/(N-1)
dy=1.0/(N-1)

P=np.zeros((N+1,N+1))
U=np.zeros((N+1,N+1))
V=np.zeros((N+1,N+1))
U_1=np.zeros((N+1,N+1))
V_1=np.zeros((N+1,N+1))
U_2=np.zeros((N+1,N+1))
V_2=np.zeros((N+1,N+1))
U_prev=np.zeros((N+1,N+1))
V_prev=np.zeros((N+1,N+1))
P_prev=np.zeros((N+1,N+1))
UC=np.zeros((N,N))
VC=np.zeros((N,N))
PC=np.zeros((N,N))

#benchmark
x_b=[0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]
y_b=[0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]
Re_100_v=[0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0]
Re_100_u=[0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1]

pt=np.zeros(N)
for i in range(1,N):
    pt[i]=(i-1)/(N-1)

def meet_poisson():
  residual=0
  for i in range(1,N):
    for j in range(1,N):
      ux=(U_2[i,j]-U_2[i-1,j])/dx
      vy=(V_2[i,j]-V_2[i,j-1])/dy
      poisson_LHP=(ux+vy)/dt
      poisson_RHP=(P[i+1,j]+P[i-1,j]+P[i,j+1]+P[i,j-1]-4*P[i,j])/(dx*dx)
      residual+=abs(poisson_LHP-poisson_RHP)

  #clear_output(wait=True)
  print("[meet Poisson]residual: ", residual)
  if(residual<tol_p):
    return True

  return False

def is_steady():
  vt=0
  for i in range(1,N):
    for j in range(1,N):
      vt+=abs(U_prev[i,j]-U[i,j])
      vt+=abs(V_prev[i,j]-V[i,j])
      U_prev[i,j]=U[i,j]
      V_prev[i,j]=V[i,j]

  print("[is_steady] velocity deviation: ",vt)
  if(vt<tol_v):
    return True

  return False

def collocate():
  for i in range(N):
    for j in range(N):
      UC[i,j]=0.5*(U[i,j]+U[i,j+1])
      VC[i,j]=0.5*(V[i,j]+V[i+1,j])
      PC[i,j]=(P[i,j]+P[i+1,j]+P[i,j+1]+P[i+1,j+1])*0.25

def moniter():
  clear_output(wait=True)
  print("Timestep: ", timestep)
  fig, ax = plt.subplots(1, 3, figsize=(17,5))
  ax[0].plot(Re_100_u, y_b, 'bo',np.transpose(UC)[:,int((N+1)/2)], pt)
  ax[0].set(ylim=(0, 1))
  ax[1].plot(x_b, Re_100_v, 'bo', pt, np.transpose(VC)[int((N+1)/2)])
  ax[1].set(xlim=(0, 1))
  ax[2].imshow(np.transpose(np.sqrt(np.multiply(UC, UC)+np.multiply(VC, VC))))
  ax[2].invert_yaxis()
  plt.show()

#Computation-1
def setBC(option):
  if option=='P':
    P[0,:]=P[1,:] #west
    P[N,:]=P[N-1,:] #east
    P[:,0]=P[:,1] #south
    P[:,N]=P[:,N-1] #north

  elif option=='U':
    U[:,N]=2-U[:,N-1] #north
    U[:,0]=-U[:,1] #south
    U[0,:]=0 #west
    U[N-1,:]=0 #east
    
    U_1[:,N]=2-U_1[:,N-1] #north
    U_1[:,0]=-U_1[:,1] #south
    U_1[0,:]=0 #west
    U_1[N-1,:]=0 #east

    U_2[:,N]=2-U_2[:,N-1] #north
    U_2[:,0]=-U_2[:,1] #south
    U_2[0,:]=0 #west
    U_2[N-1,:]=0 #east
    
  elif option=='V':
    V[0,:]=-V[1,:] #west
    V[N,:]=-V[N-1,:] #east
    V[:,0]=0 #south
    V[:,N-1]=0 #north

    V_1[0,:]=-V_1[1,:] #west
    V_1[N,:]=-V_1[N-1,:] #east
    V_1[:,0]=0 #south
    V_1[:,N-1]=0 #north

    V_2[0,:]=-V_2[1,:] #west
    V_2[N,:]=-V_2[N-1,:] #east
    V_2[:,0]=0 #south
    V_2[:,N-1]=0 #north

def solve_U1():
  for i in range(1,N-1):
    for j in range(1,N):
      u=U[i,j]
      v=(V[i,j]+V[i+1,j]+V[i,j-1]+V[i+1,j-1])/4
      ux=(U[i+1,j]-U[i-1,j])/(2*dx)
      uy=(U[i,j+1]-U[i,j-1])/(2*dy)
      u2x=(U[i+1,j]+U[i-1,j]-2*U[i,j])/(dx*dx)
      u2y=(U[i,j+1]+U[i,j-1]-2*U[i,j])/(dy*dy)

      C=u*ux+v*uy
      D=(u2x+u2y)/Re

      px=(P[i+1,j]-P[i,j])/dx
      U_1[i,j]=(-C+D-px)*dt+U[i,j]

def solve_V1():
  for i in range(1,N):
    for j in range(1,N-1):
      u=(U[i-1,j+1]+U[i,j+1]+U[i-1,j]+U[i,j])/4.0;
      v=V[i,j]
      vx=(V[i+1,j]-V[i-1,j])/(2*dx)
      vy=(V[i,j+1]-V[i,j-1])/(2*dy)
      v2x=(V[i+1,j]+V[i-1,j]-2*V[i,j])/(dx*dx)
      v2y=(V[i,j+1]+V[i,j-1]-2*V[i,j])/(dy*dy)
      
      C=u*vx+v*vy
      D=(v2x+v2y)/Re
      
      py=(P[i,j+1]-P[i,j])/dy
      V_1[i][j]=(-C+D-py)*dt+V[i,j]

def solve_U2():
  for i in range(1,N-1):
    for j in range(1,N):
      px=(P[i+1,j]-P[i,j])/dx
      U_2[i,j]=px*dt+U_1[i,j]

def solve_V2():
  for i in range(1,N):
    for j in range(1,N-1):
      py=(P[i,j+1]-P[i,j])/dy
      V_2[i,j]=py*dt+V_1[i,j]

def solve_P():
  iteration=0
  while(meet_poisson()==False):
    iteration=iteration+1
    #clear_output(wait=True)
    #print("[solve_P]iteration: ", iteration)
    for i in range(1,N):
      for j in range(1,N):
        ux=(U_2[i,j]-U_2[i-1,j])/dx
        vy=(V_2[i,j]-V_2[i,j-1])/dy
        poisson_LHP=(ux+vy)/dt
        P[i,j]=0.25*(P[i+1,j]+P[i-1,j]+P[i,j+1]+P[i,j-1]-poisson_LHP*dx*dx)
  meet_poisson()

def solve_U():
  for i in range(1,N-1):
    for j in range(1,N):
      px=(P[i+1,j]-P[i,j])/dx
      U[i,j]=-px*dt+U_2[i,j]

def solve_V():
  for i in range(1,N):
    for j in range(1,N-1):
      py=(P[i,j+1]-P[i,j])/dy
      V[i,j]=-py*dt+V_2[i,j]

#Computation-2
def set_BC(option=3):
  if option==0:
    setBC('P')

  elif option==1:
    setBC('U')

  elif option==2:
    setBC('V')
    
  elif option==3:
    setBC('U')
    setBC('V')
    setBC('P')

def step_1(option):
  if option==1:
    solve_U1()
  elif option==2:
    solve_V1()

def step_2(option):
  if option==1:
    solve_U2()
  elif option==2:
    solve_V2()

def step_3(option):
  solve_P()

def step_4(option):
  if option==1:
    solve_U()
  elif option==2:
    solve_V()



#Algorithm
timestep=0
set_BC()
while((timestep==0)|(is_steady()==False)):
  timestep+=1
  step_1(1)
  step_1(2)

  step_2(1)
  step_2(2)

  set_BC(1)
  set_BC(2)

  step_3(0)
  set_BC(0)

  step_4(1)
  step_4(2)
  set_BC(1)
  set_BC(2)

  collocate()
  
  moniter()