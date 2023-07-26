#################################
# buckled_psi6.py
# Seamus Gallagher
# Carnegie Mellon University
#################################

import sys
import numpy as np  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps
import vtfstream
import random
from scipy.spatial import Voronoi, voronoi_plot_2d

Ls = float(sys.argv[1])
Lx = float(sys.argv[2])
dx = Lx/1000
Ly = float(sys.argv[3])
Lz = 100
start = -100
stop = -1
interval = 1


traj_it = vtfstream.load("trajectory.vtf",start=start,stop=stop,aslipids=True) # load generator
step, box = next(traj_it)

Ntest = len(step) # number of lipids to calculate the p6 for
def nn(ind,points,Nneighbors=6): # trial lipid index, step data, n neighbors
    r0 = points[ind]
    neighbors = [] # store tuples (index,distance)
    for i,r1 in enumerate(points):
        if i == ind:
            continue
        else:
            r = np.abs(r1 - r0)
            if r[0] > Lx/2:
                r[0] = r[0] - Lx
            if r[1] > Ly/2:
                r[1] = r[1] - Ly
            r = np.linalg.norm(r)
            if len(neighbors) < Nneighbors:
                neighbors.append((i,r))
            elif any([r < x[1] for x in neighbors]):
                neighbors = sorted(neighbors,key=lambda x: x[1])
                neighbors.pop()
                neighbors.append((i,r))
            else:
                continue

    return [x[0] for x in neighbors] # just return indices

def fit_spline(leaflet1,leaflet2,box):
    # construct surface grid
    N = 50
    h1 = np.zeros(N)
    c1 = np.zeros(N)
    h2 = np.zeros(N)
    c2 = np.zeros(N)

    for lipid in leaflet1:
        x_ind = int(lipid[0]/(box[0]/N))
        h1[x_ind] += lipid[2]
        c1[x_ind] += 1

    for lipid in leaflet2:
        x_ind = int(lipid[0]/(box[0]/N))
        h2[x_ind] += lipid[2]
        c2[x_ind] += 1

    h1[c1==0] = np.nan
    h2[c2==0] = np.nan 

    spline_z = (h1/c1 + h2/c2)/2
    
    right_img = np.array([np.array(x) + np.array([Lx,0,0]) for x in list(step) if x[0] < Lx/8])
    left_img = np.array([np.array(x) - np.array([Lx,0,0]) for x in list(step) if x[0] > 7*Lx/8])
    
    PBC_skin = int(N/10) # repeat bins for spline fitting
    
    spline_x = np.linspace(-PBC_skin*(box[0]/N),box[0] + PBC_skin*(box[0]/N), N + 2*PBC_skin)
    spline_z = np.concatenate((spline_z[-PBC_skin:],spline_z,spline_z[:PBC_skin]))

    cs = UnivariateSpline(np.sort(spline_x),spline_z[np.argsort(spline_x)],s=1) # initial guess for variances
    dcsdx = cs.derivative()
    Ls = simps([np.sqrt(1 + dcsdx(x)**2) for x in np.arange(0,Lx,dx)],x=np.arange(0,Lx,dx))
    
    return cs,dcsdx,Ls

def project_points(points,cs,dcsdx):
    proj_points = []
    for point in points: # only project one leaflet
        p = np.array([point[0]%Lx,point[2]%Lz])
        # Hossein and Deserno Stiffening transition in asymmetric lipid bilayers: The role of highly ordered domains and the effect of temperature and size
        x = p[0] + (dcsdx(p[0])/(1+dcsdx(p[0])**2))*(p[1]-cs(p[0]))
        x = x%Lx
        arc_int = [np.sqrt(1 + dcsdx(u)**2) for u in np.arange(0,x,dx)] # arc length integrand
        s = simps(arc_int,x=np.arange(0,x,dx))
        s = s%Ls
        proj_points.append(np.array([s,point[1]]))
    return np.array(proj_points)
    
def frame_p6(points): # Takes list of points [[x1,y1,...],[x2,y2,...],...] ; Gives list of points with p6 as z-coordinate [[x1,y2,p61],...]
    p6_list = []
    for i,ri in enumerate(points):
        neighbors = nn(i,points)
        p6 = 0
        
        for j in neighbors:
            rj = points[j]
            r_ij = rj - ri
            
            if r_ij[0] > Ls/2:
                r_ij[0] = r_ij[0] - Ls
            if r_ij[0] <= -Ls/2:
                r_ij[0] = r_ij[0] + Ls
            if r_ij[1] > Ly/2:
                r_ij[1] = r_ij[1] - Ly
            if r_ij[1] <= -Ls/2:
                r_ij[1] = r_ij[1] + Ly

            theta_ij = np.arccos(np.dot(r_ij,np.array([0,1]))/np.linalg.norm(r_ij))
            p6 += 1/6*np.exp(6.0j*theta_ij)
        p6_list.append(np.array([ri[0],ri[1],p6]))
        
    return p6_list

def voronoi_areas(proj_points,areas):
    vor = Voronoi(proj_points)
    vertices = vor.vertices
    regions = vor.regions

    for region in regions:
        if region.count(-1) > 0: continue # scipy voronoi uses -1 index to indicate boundary
        A = 0
        for i_j,j in enumerate(region):
            # Surveyor's formula
            k = region[(i_j+1)%len(region)]
            A += 0.5*(vertices[j,0] + vertices[k,0])*(-vertices[j,1]+vertices[k,1])
        areas.append(abs(A))

    return areas

areas = []
i = start
while i < stop:
    print(i)
    if i % interval == 0:
        try:
            
            step, box = next(traj_it)
            
            step = np.array(step)
            leaflet1 = np.mean(step[:int(len(step)/2),1:],axis=1)%box # take last tail positions
            leaflet2 = np.mean(step[int(len(step)/2):,1:],axis=1)%box
            step = np.array(leaflet1 + leaflet2)

            print("Fitting spline") # fit buckle to cubic spline in XZ plane
            cs,dcsdx,Ls = fit_spline(leaflet1,leaflet2,box)
            
            print("Projecting lipids") # project lipids x and z coordinates onto the spline
            proj_points = project_points(points,cs,dcsdx)
             
            print("Calculating p6")
            p6 = frame_p6(proj_points)
            np.save(str(i)+'_p6.npy',p6) # SHOULD BE CAREFUL TO CHANGE THIS IF YOURE GOING TO DO A LOT OF FRAMES

            areas = voronoi_areas(proj_points,areas)

        except StopIteration:
            print("Error: out of tajectory data in process "
                  + f"with start and end indices {indices[0]} and {indices[-1]}")
            print(f"i={i}, and ind={ind}")
            raise
    i += 1
np.save('areas.npy',areas)

