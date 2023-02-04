#!/usr/bin/python2
# Copyright (C) 2015 Universite de Geneve, Switzerland
# E-mail contact: jonas.latt@unige.ch

from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
from pyevtk.hl import gridToVTK  

### before start...
###   pip install pyevtk
###   make dir pngfiles
###   make dir vtkfiles

###### Flow definition #########################################################
maxIter = 100          # total time iterations.
Re = 20.0              # Reynolds number.
nx,ny,nz = 200,100,100 # number of lattice nodes.
cx = nx//4             # cx cylinder coords
cy = ny//2             # cy
cz = nz//2             # cz
cr = ny//8             # cr cylinder radius
uLB   = 0.04           # velocity in lattice units.
nulb  = uLB*cr/Re      # viscoscity in lattice units.
omega = 1/(3*nulb+0.5) # relaxation parameter.
writeout = 1           # text output frequency
pngsave = 100          # png save frequency 
vtksave = 100          # vtk save frequency
###### Lattice Constants #######################################################
v = array([  \
 [1,0,0],[0,1,0],[0,0,1],[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[0,0,0], \
 [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[0,0,-1],[0,-1,0],[-1,0,0] ])

t = array([ \
1/9,1/9,1/9,1/72,1/72,1/72,1/72, 2/9, 1/72,1/72,1/72,1/72,1/9,1/9,1/9])

col1 = array([0, 3, 4, 5, 6])
col2 = array([1, 2, 7,12,13])
col3 = array([8, 9,10,11,14])
###### Function Definitions ####################################################
def macroscopic(fin): # 
    rho = sum(fin, axis=0)
    u = zeros((3,nx,ny,nz))
    for i in range(15):
        u[0] += v[i,0] * fin[i]
        u[1] += v[i,1] * fin[i]
        u[2] += v[i,2] * fin[i]
    u = u/rho
    return rho, u

def equilibrium(rho, u): # equilibrium distribution function
    usqr = u[0]**2 + u[1]**2 + u[2]**2
    feq = zeros((15,nx,ny,nz))
    for i in range(15):
        cu = v[i,0]*u[0] + v[i,1]*u[1] + v[i,2]*u[2]
        feq[i] = rho*t[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*usqr)
    return feq

def obstacle_fun(x, y, z):
    return (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < cr**2

obstacle = fromfunction(obstacle_fun,(nx,ny,nz))

def inivel(d, x, y, z):
    return 1//(d+1) * uLB * (1 + 1e-4*sin(y/(ny-1)*2*pi))*(1 + 1e-4*sin(z/(nz-1)*2*pi))

vel = fromfunction(inivel,(3,nx,ny,nz))

fin = equilibrium(1, vel) # Initialize populations at equilibrium

###### Main time loop ##########################################################

for time in range(maxIter+1):

    # Right wall: outflow condition.
    fin[col3,-1,:,:] = fin[col3,-2,:,:] 

    # Compute macroscopic variables, density and velocity.
    rho, u = macroscopic(fin)

    # Left wall: inflow condition.
    u[:,0,:,:] = vel[:,0,:,:]
    rho[0,:,:] = 1/(1-u[0,0,:,:]) * ( sum(fin[col2,0,:,:], axis=0) \
                                  + 2*sum(fin[col3,0,:,:], axis=0) )
    # Compute equilibrium.
    feq = equilibrium(rho, u)
    fin[[0, 3, 4, 5,6],0,:,:] = feq[[0, 3, 4, 5,6],0,:,:] + \
    fin[[14,11,10,9,8],0,:,:] - feq[[14,11,10,9,8],0,:,:]

    # Collision step.
    fout = fin - omega * (fin - feq)

    # Bounce-back condition for obstacle.
    for i in range(15):
        fout[i, obstacle] = fin[14-i, obstacle]

    # Streaming step.
    for i in range(15):
         fin[i] = roll( roll( roll( fout[i], \
         v[i,0],axis=0), v[i,1],axis=1), v[i,2],axis=2)
    
    # write to screen
    if (time%writeout==0):
        print(time)
        
    # visualization of the velocity
    if (time%pngsave==0)&(time>0):
        print ("image_{0:04d}.png".format(time//pngsave))
        plt.figure(figsize = (10,5))
        data1 = sqrt( u[0]**2 + u[1]**2 + u[2]**2 )
        data2 = data1[:,:,cz]*(1-obstacle[:,:,cz])
        plt.imshow(data2.transpose(),cmap="gist_ncar") #"rainbow")
        plt.grid(True)
        plt.clim(0.02,0.06) 
        plt.colorbar(shrink=0.75,format='%.6f')
        plt.savefig("./pngfiles/3Dimage1_{0:04d}.png".format(time//pngsave))
        plt.close()        

    # save VTK flow solution
    if (time%vtksave==0)&(time>0):
        print ("lbsol_{0:04d}.vtr".format(time//pngsave))
        x = arange(0, nx+1)
        y = arange(0, ny+1)
        z = arange(0, nz+1)
        data = sqrt( u[0]**2 + u[1]**2 + u[2]**2 )
        dataStacked = dstack([data])
        gridToVTK("./vtkfiles/lbmsol_{0:04d}".format(time//vtksave),x,y,z,cellData={'data':dataStacked})

