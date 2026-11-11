# Aero 421 final spacecraft redo (python)
import math
import numpy as np

#Functions
def ecross(r):#cross operation 
    ecros = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return ecros
def inertia(mass,dim,r): # Note this r is distance from origin minus com distance.
    Ixx = 1/12*mass*(dim[1]**2+dim[2]**2) #could simplify to 1/6*mass*side^2 for future reference
    Iyy = 1/12*mass*(dim[0]**2+dim[2]**2)
    Izz = 1/12*mass*(dim[0]**2+dim[1]**2)
    J = np.array([Ixx-mass*(ecross(r)@ecross(r)), Iyy-mass*ecross(r)@ecross(r),Izz-mass*(ecross(r)@ecross(r))])
    return J

     
#Main code
def main():
#Part 1
    m_tot = 640 #[kg]
    m_bus = 500 #[kg]
    m_sensor = 100 # [kg]
    m_panels = 20 # [kg] There are 2x panels each 20 kg.
    m_total = m_bus + m_sensor + 2*m_panels
    print(f"total mass: {m_total}")

    dim_bus = np.array([2, 2, 2]) # [m] 2 meter cube
    com_bus = dim_bus/2 - dim_bus/2

    x_com = com_bus[0]*m_total/m_total
    y_com = com_bus[1]*m_total/m_total
    z_com = com_bus[2]*m_total/m_total
    com_tumble = np.array([x_com, y_com, z_com])
    print(f'Center of mass for bus {com_tumble} [m]')

    dim_panel = np.array([2, 3, .05]) #Panel dimensions [m]
    com_panels = dim_panel - dim_panel

    dim_sensor = np.array([.25, .25, 1])
    #Center of mass for panels based on dimensions on the spacecraft (given)
    com_sensor = np.array([dim_sensor[0]/2, dim_sensor[1]/2, dim_sensor[2]]) - np.array([dim_sensor[0]/2, dim_sensor[1]/2, 0]) + np.array([0, 0, .5])

    xt_com = (m_bus*com_bus[0]+ (m_panels*com_panels[0]) + m_sensor*com_sensor[0])/m_total
    yt_com = (m_bus*com_bus[1]+ (m_panels*com_panels[1]) + m_sensor*com_sensor[1])/m_total
    zt_com = (m_bus*com_bus[2]+ (m_panels*com_panels[2]) + m_sensor*com_sensor[2])/m_total
    com_total = np.array([xt_com, yt_com, zt_com])
    print(f"Center of mass normal:{com_total}")

    #Distance from origin
    r0_bus = np.array([0, 0, 0])
    r0_panel1 = np.array([0, -2.5, 0])
    r0_panel2 = np.array([0,  2.5, 0])
    r0_sensor = np.array([0, 0, 1.5])

    #relative Center of mass 
    r_bus = r0_bus - com_total
    r_panel1 = r0_panel1 - com_total
    r_panel2 = r0_panel2 - com_total
    r_sensor = r0_sensor - com_total
    
    #Inertia for bus
    J_bus = inertia(m_bus, dim_bus, r_bus) # 3, 3x3 matrices one per I__ direction
    print(J_bus) 
    
    #Inertia for panels
    J_panel1 = inertia(m_panels, dim_panel, r_panel1)
    J_panel2 = inertia(m_panels, dim_panel, r_panel2)
    print(J_panel1)

    #Inertia for sensor
    J_sensor = inertia(m_sensor, dim_sensor, r_sensor)

    #Inertial matrix total 
    J_xx = J_bus[0,0]+J_panel1[0,0]+J_panel2[0,0]+J_sensor[0,0]
    J_yy = J_bus[1,0]+J_panel1[1,0]+J_panel2[1,0]+J_sensor[1,0]
    J_zz = J_bus[2,0]+J_panel1[2,0]+J_panel2[2,0]+J_sensor[2,0]
    J_normal = np.array([J_xx[0,0],J_yy[1,1],J_zz[2,2]])
    print(J_normal)


if __name__ == "__main__":
    main()

