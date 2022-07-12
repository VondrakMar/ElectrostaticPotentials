import numpy as np
from scipy.special import erf
from ase.units import Hartree, Bohr

def readData(name):
    data = []
    with open(name,"r") as file:
        for line in file:
            line = line.split()
            data.append({"xyz":[float(line[0]),float(line[1]),float(line[2])],"q":float(line[3]),"a":float(line[4])})
    return data

def rewrite_vmd(data):
    f = open("vmdfile.xyz","w")
    f.write(str(len(data))+"\n")
    f.write("random comment\n")
    for a in data:
        f.write(f"H {str(a['xyz'][0])} {str(a['xyz'][1])} {str(a['xyz'][2])}\n")   
    f.close()


def points(data,i,j):
    xyz1 = np.array(data[i]["xyz"])
    xyz2 = np.array(data[j]["xyz"])
    q1 = data[i]["q"]
    q2 = data[j]["q"]
    distance = np.linalg.norm(np.subtract(xyz1,xyz2))
    return (q1*q2/distance)

def gaussians(data,i,j):
    xyz1 = np.array(data[i]["xyz"])
    xyz2 = np.array(data[j]["xyz"])
    q1 = data[i]["q"]
    q2 = data[j]["q"]
    alpha1 = data[i]["a"]
    alpha2 = data[j]["a"]
    distance = np.linalg.norm(np.subtract(xyz1,xyz2))
    return (q1*q2*erf(np.sqrt(alpha1**2+alpha2**2)*distance)/distance)

def points_all(data):
    xyz = []
    qs = []
    for a in data:
        xyz.append(a["xyz"])
        qs.append(a["q"])
    print("charge in the cell:",np.sum(qs))
    potential = 0
    for main in range(len(xyz)):
        for main2 in range(len(xyz)):
            if main != main2:
                distance = calc_Rvec_periodic(np.array(xyz[main])/Bohr,np.array(xyz[main2])/Bohr)
                potential += qs[main]*qs[main2%len(xyz)]/distance
    return 0.5*potential*Hartree

def points_all_force(data):
    xyz = []
    qs = []
    forces = []
    for atom in data:
        forces.append([0,0,0])
    for a in data:
        xyz.append(a["xyz"])
        qs.append(a["q"])
    print("charge in the cell:",np.sum(qs))
    potential = 0
    for main in range(len(xyz)):
        for main2 in range(len(xyz)):
            if main != main2:
                delx = xyz[main][0]/Bohr - xyz[main2][0]/Bohr
                dely = xyz[main][1]/Bohr - xyz[main2][1]/Bohr
                delz = xyz[main][2]/Bohr - xyz[main2][2]/Bohr
                rsq = delx*delx + dely*dely + delz*delz
                r2inv = 1.0/rsq
                rinv = np.sqrt(r2inv)
                forcecoul = qs[main]*qs[main2]*rinv
                fpair = forcecoul * r2inv
                forces[main][0] += delx*fpair*Hartree/Bohr
                forces[main][1] += dely*fpair*Hartree/Bohr
                forces[main][2] += delz*fpair*Hartree/Bohr
    return forces


def gaussians_all_force(data):
    xyz = []
    qs = []
    alphas = []
    for a in data:
        xyz.append(a["xyz"])
        qs.append(a["q"])
        alphas.append(a["a"])
    forces = []
    for atom in data:
        forces.append([0,0,0])
    print("charge in the cell:",np.sum(qs))
    potential = 0
    for main in range(len(xyz)):
        dVdr = 0
        for main2 in range(len(xyz)):
            dVdr = 0
            if main != main2:
                delx = xyz[main][0]/Bohr - xyz[main2][0]/Bohr
                dely = xyz[main][1]/Bohr - xyz[main2][1]/Bohr
                delz = xyz[main][2]/Bohr - xyz[main2][2]/Bohr
                rsq = delx*delx + dely*dely + delz*delz
                r2inv = 1.0/rsq
                rinv = np.sqrt(r2inv)
                r = np.sqrt(rsq)
                ga_ij = 1/np.sqrt(alphas[main]**2+alphas[main2]**2)
                dVdr += 2.0*ga_ij*np.exp(-ga_ij**2 * r**2)/(np.sqrt(np.pi)*r)
                dVdr += -erf(ga_ij*r)/(r**2)
                forces[main][0] += -delx*dVdr*qs[main]*qs[main2]*rinv*Hartree/Bohr
                forces[main][1] +=  -dely*dVdr*qs[main]*qs[main2]*rinv*Hartree/Bohr
                forces[main][2] +=  -delz*dVdr*qs[main]*qs[main2]*rinv*Hartree/Bohr
    return forces

def gaussians_all(data):
    xyz = []
    qs = []
    alphas = []
    for a in data:
        xyz.append(a["xyz"])
        qs.append(a["q"])
        alphas.append(a["a"])
    print("charge in the cell:",np.sum(qs))
    potential = 0
    for main in range(len(xyz)):
        for main2 in range(len(xyz)):
            if main != main2:
                distance = calc_Rvec_periodic(np.array(xyz[main])/Bohr,np.array(xyz[main2])/Bohr)
                potential += qs[main]*qs[main2]*erf(np.sqrt((alphas[main])**2+(alphas[main2]**2))*distance)/distance
    return 0.5*potential*Hartree


def point_periodic(data):
    xyz = []
    qs = []
    for a in data:
        xyz.append(a["xyz"])
        qs.append(a["q"])
    print("charge in the cell:",np.sum(qs))
    period_xyz = create_pbc(data,3,3,3)
    potential = 0
    for main in range(len(xyz)):
        for periodic in range(len(period_xyz)):
            if main != periodic:
                distance = calc_Rvec_periodic(np.array(xyz[main])*Bohr,np.array(period_xyz[periodic])*Bohr)
                #print(potential)
                potential += qs[main]*qs[periodic%len(xyz)]/distance
    return potential

def gaussian_periodic(data):
    xyz = []
    qs = []
    alphas = []
    for a in data:
        xyz.append(a["xyz"])
        qs.append(a["q"])
        alphas.append(a["a"])
    print("charge in the cell:",np.sum(qs))
    period_xyz = create_pbc(data,3,3,3)
    potential = 0
    potPoint = 0
    for main in range(len(xyz)):
        for periodic in range(len(period_xyz)):
            if main != periodic:
                distance = calc_Rvec_periodic(xyz[main],period_xyz[periodic])
                glo = qs[main]*qs[periodic%len(xyz)]*erf(np.sqrt(alphas[main]**2+alphas[periodic%len(xyz)]**2)*distance)/distance
                potential += glo
                potPoint += qs[main]*qs[periodic%len(xyz)]/distance
                #print(potential,potPoint)
                #print("G:",glo)
                #print("point:",qs[main]*qs[periodic%len(xyz)]/distance)
                #print(glo - qs[main]*qs[periodic%len(xyz)]/distance)
    return potential


def create_pbc(data,l1,l2,l3):
    xyz = []
    for a in data:
        xyz.append(a["xyz"])
    xyz_periodic = xyz.copy()
    dims = np.array([l1,l2,l3])
    dimension_matrix = np.eye(3)*dims
    for x in [-3,-2,-1,0,1,2,3]:
        for y in [-3,-2,-1,0,1,2,3]:
            for z in [-3,-2,-1,0,1,2,3]:
#    for x in [-1,0,1]:
#        for y in [-1,0,1]:
#            for z in [-1,0,1]:
                if x == 0 and y == 0 and z == 0:
                    continue
                xyz_periodic=np.append(xyz_periodic,(xyz + x*dimension_matrix[0] + y*dimension_matrix[1]+ z*dimension_matrix[2]),axis=0)
    return xyz_periodic
    

def calc_Rvec_periodic(xyz1,xyz2,alpha=15, beta = 30):
    #xyz1 = np.array(])
    #xyz2 = np.array(data[j]["xyz"])
    r_temp = np.linalg.norm(np.subtract(xyz1,xyz2))
    if r_temp <= alpha:
        r = r_temp
    elif r_temp > alpha and r_temp <= beta:
        r = (alpha*alpha)/(2*(alpha-beta))+r_temp*(1-(alpha)/((alpha-beta)))+(r_temp*r_temp)/(2*(alpha-beta))
    elif r_temp > beta:
        r = beta + ((alpha*alpha+beta*beta)-2*alpha*beta)/(2*(alpha-beta))
    return r
