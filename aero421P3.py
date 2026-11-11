################## Aero 421 sc part 1---3 ########################
import math
import numpy as np
import matplotlib
matplotlib.use("QtAgg") # Need this to run my python plots on linux 
import matplotlib.pyplot as plt
plt.style.use("dark_background")

###############Quality of life##################
sin =np.sin
asin = np.asin
cos =np.cos
tan = np.tan
sec = lambda x: 1/cos(x)
pi = np.pi
sqrt = math.sqrt
inv = np.linalg.inv
cross = np.linalg.cross
norm = np.linalg.norm
atan = math.atan
atan2 = math.atan2
trace = np.trace
zeros = np.zeros # Need (()) double parenthesis for colunm vectors
solve = np.linalg.solve
cross = np.cross
eye = np.eye
deg2rad = np.deg2rad
rad2deg = np.rad2deg
ones = np.ones
exp = np.exp


########################### Functions ####################################### 

def ecross(r):#cross operation 
    ecros = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]]) #Ensure its 3x3 matrix 1x3
    return ecros

def ecros(r):
    ecros = np.array([[0, -r[2,0], r[1,0]], [r[2,0], 0, -r[0,0]], [-r[1,0], r[0,0], 0]]) #This is for 3x1
    return ecros

def inertia(mass,dim,r): # Note this r is distance from origin minus com distance.
    Ixx = 1/12*mass*(dim[1]**2+dim[2]**2) #could simplify to 1/6*mass*side^2 for future reference (for cube only)
    Iyy = 1/12*mass*(dim[0]**2+dim[2]**2)
    Izz = 1/12*mass*(dim[0]**2+dim[1]**2)
    Icom = np.diag([Ixx,Iyy,Izz])
    J = Icom - mass*(ecross(r)@ecross(r))
    return J

def Cz(psi):
    R = np.array([[cos(psi),-sin(psi), 0],[sin(psi), cos(psi), 0], [0, 0, 1]])
    return R

def Cx(phi):
    R = np.array([[1, 0, 0,], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
    return R


def coe2rv(h, ecc, inc, omega, raan, TA, mu): # This specifically finds perifocal r and v and then transforms to ECI r and v
    r_peri = (h**2/mu) * (1/(1+ecc*cos(TA))) * np.array([cos(TA), sin(TA), 0])
    v_peri = (mu/h) * np.array([-sin(TA), ecc + cos(TA), 0])
    R3_RAAN = Cz(raan) #omega 
    R1_inc = Cx(inc) # theta
    R3_omega = Cz(omega) # angular omega (w)
    Q = R3_RAAN@R1_inc@R3_omega

    #This is for ECI to peri
    #r = (Q).T @ r_peri 
    #v = (Q).T @ v_peri

    #Dont need transpose since going from peri to ECI
    r = Q @ r_peri
    v = Q @ v_peri


    rv = np.array([r,v])
    return rv

def rot313(phi,theta,psi):
    R3_RAAN = Cz(phi)
    R1_inc = Cx(theta)
    R3_omega = Cz(psi)
    Q = R3_RAAN@R1_inc@R3_omega
    return Q

def dynamics(T,J,omega):
    domega = inv(J)@(T-ecross(omega)@J@omega) 
    return domega 

def rot2eul(C):
    E1 = atan2(C[1,2], C[2,2])
    E2 = -asin(C[0,2])
    E3 = atan2(C[0,1], C[0,0])
    E = np.array([[E1],[E2],[E3]])
    return E

def rot2quat(C):
    eta = sqrt(trace(C)+1)/2
    # eps = zeros(3)
    eps0 = (C[1,2]-C[2,1])/(4*eta) 
    eps1 = (C[2,0]-C[0,2])/(4*eta)
    eps2 = (C[0,1]-C[1,0])/(4*eta)
    q = np.array([[eps0],[eps1],[eps2],[eta]])
    return q 

#def integrate(x, dx, dt): #basic integrator when you have value and its derivative
    #return x+dx*dt 

def quat2C21(q):
    eps = q[0:3]
    eta = q[3]
    C21 = (2*eta**2 -1)*eye(3) + 2*eps@eps.T - 2*eta*ecros(eps)
    return C21


    
########################################## Main code ###############################################
def main():
#Part 1
    m_tot = 640 #[kg]
    m_bus = 500 #[kg]
    m_sensor = 100 # [kg]
    m_panels = 20 # [kg] There are 2x panels each 20 kg.
    m_total = m_bus + m_sensor + 2*m_panels
    print(f"total mass: \n {m_total} \n")

    dim_bus = np.array([2, 2, 2]) # [m] 2 meter cube
    com_bus = dim_bus/2 - dim_bus/2

    # Center of mass for bus
    x_com = com_bus[0]*m_total/m_total
    y_com = com_bus[1]*m_total/m_total
    z_com = com_bus[2]*m_total/m_total
    com_tumble = np.array([x_com, y_com, z_com])

    dim_panel = np.array([2, 3, .05]) #Panel dimensions [m]
    com_panels = dim_panel - dim_panel # Center of mass for both panels 

    dim_sensor = np.array([.25, .25, 1])
    #Center of mass for sensors based on dimensions on the spacecraft (given)
    com_sensor = np.array([dim_sensor[0]/2, dim_sensor[1]/2, dim_sensor[2]]) - np.array([dim_sensor[0]/2, dim_sensor[1]/2, 0]) + np.array([0, 0, .5])

    xt_com = (m_bus*com_bus[0]+ (m_panels*com_panels[0]) + m_sensor*com_sensor[0])/m_total
    yt_com = (m_bus*com_bus[1]+ (m_panels*com_panels[1]) + m_sensor*com_sensor[1])/m_total
    zt_com = (m_bus*com_bus[2]+ (m_panels*com_panels[2]) + m_sensor*com_sensor[2])/m_total
    com_total = np.array([xt_com, yt_com, zt_com])
    print(f"Center of mass normal: \n {com_total}\n")

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
    
    #Inertia for panels
    J_panel1 = inertia(m_panels, dim_panel, r_panel1)
    J_panel2 = inertia(m_panels, dim_panel, r_panel2)

    #Inertia for sensor
    J_sensor = inertia(m_sensor, dim_sensor, r_sensor)

    #Inertial matrix total 
    J_normal = J_bus + J_panel1 + J_panel2 + J_sensor 
    print(f'Normal operation torque: \n {J_normal} \n')

################################# Start of part 2 ###################################################
    JD_0 = 2460390
    m_b = np.array([[0], [0], [-.5]])
    mu = 398600
    h = 53335.2
    ecc = 0
    RAAN = deg2rad(0)
    inc = deg2rad(98.43)
    omega = deg2rad(0)
    TA = deg2rad(0)#nu
    a = h**2/mu/(1 - ecc**2)
    t = 2*pi*sqrt(a**3/mu) # orbital period
    T = np.array([[0],[0],[0]])
    revs = 5

    # Pulling R and V vectors in ECI and converting them to LVLH
    [r_eci,v_eci] = coe2rv(h, ecc, inc, omega, RAAN, TA, mu)
    y_LVLH = -cross(r_eci,v_eci) / norm(cross(r_eci,v_eci))
    z_LVLH = -r_eci / norm(r_eci)
    x_LVLH = cross(y_LVLH,z_LVLH)
    c_LVLH_ECI = np.array([x_LVLH,y_LVLH,z_LVLH])

    # initials
    phi_0 = 0
    theta_0 = 0 
    psi_0 = 0 

    Eul_b_LVLH0 = np.array([[phi_0],[theta_0],[psi_0]]) #Initial euler angles

    quat_b_LVLH0 = np.array([[0],[0], [0], [1]]) #INitial quaternion values 
    
    print(f'Rotation matrix relating LVLH & ECI \n {c_LVLH_ECI} \n')

    c_ECI_LVLH = c_LVLH_ECI.T
    #print(c_ECI_LVLH)

    c_b_LVLH0 = np.eye(3) # initial LVLH-ECI

    c_b_ECI0 = c_LVLH_ECI # Rotation matrix same as for LVLH to ECI 

    e_b_ECI0 = rot2eul(c_b_ECI0)
    print(f'euler \n{e_b_ECI0[1,0]} \n')

    q_b_ECI0 = rot2quat(c_b_ECI0)
    print(f'quaternion \n {q_b_ECI0} \n')

    # Initial body rates given of body to ECI 
    w_b_ECI0 = np.array([[0.001],[-0.001],[0.002]])
    

    #simulink portion translated to python

    # need to find time out, w_b_eci, q_b_eci, e_b_eci over time

    # domega = solve(J_normal, T-np.cross(w_b_ECI0,J_normal@w_b_ECI0)) not using this one since being funky need (3,) vector but thats 
    # confusing so just ecross(omega) does same thing

    # preallocating my values used for iterations 
    dt = 1
    
    time_history = []
    w_b_ECI_history = []

    q_b_ECI_history = []

    e_b_ECI_history = []
    
    dq = zeros((4,1))
    
    #time = int(t)
    
    for i in range(6001): # this part can be optimized in next part for now leaving like this
        t_i = i * dt
        time_history.append(t_i)

        domega = solve(J_normal, T - ecros(w_b_ECI0)@(J_normal@w_b_ECI0))
        w_b_ECI0 = w_b_ECI0 + domega*dt
        w_b_ECI_history.append(w_b_ECI0.copy())

        eps = q_b_ECI0[0:3]
        eta = q_b_ECI0[3]
        dq[0:3] = 1/2*(eta*eye(3) + ecros(q_b_ECI0))@w_b_ECI0
        dq[3] = -1/2*eps.T@w_b_ECI0

        q_b_ECI0 = q_b_ECI0 + dq*dt
        q_b_ECI_history.append(q_b_ECI0.copy())

        e_phi = e_b_ECI0[0,0]
        e_theta = e_b_ECI0[1,0]
        e_psi = e_b_ECI0[2,0] 
        de = np.array([[1, sin(e_phi)*tan(e_theta), cos(e_phi)*tan(e_theta)],[ 0, cos(e_phi), -sin(e_phi)], [0, sin(e_phi)*sec(e_theta), cos(e_phi)*sec(e_theta)]])@w_b_ECI0 # 3-2-1 rotation times angular velocity

        e_b_ECI0 = e_b_ECI0 + de*dt
        e_b_ECI_history.append(e_b_ECI0.copy())

    w_hist = np.array(w_b_ECI_history)
    time = np.array(time_history)

    w_x = w_hist[:,0]
    w_y = w_hist[:,1]
    w_z = w_hist[:,2] 

    q_hist = np.array(q_b_ECI_history)

    eps1 = q_hist[:,0]
    eps2 = q_hist[:,1]
    eps3 = q_hist[:,2] 
    eta = q_hist[:,3]
    
    e_hist = np.array(e_b_ECI_history) # this spits out values in radians gotta make sure to convert to degrees

    e1 = rad2deg(e_hist[:,0])
    e2 = rad2deg(e_hist[:,1])
    e3 = rad2deg(e_hist[:,2]) 


    # Note for self, when plotting normally you can do plt.ylim for normal plot, for subplots need to add set_ylim
    
    fig, (w_plt,q_plt,e_plt) = plt.subplots(3,1)
    fig.suptitle('Free Rotation Dynamics & Kinematics')
    w_plt.plot(time, w_x,label=r'$\omega_x$')
    w_plt.plot(time,w_y,label=r'$\omega_y$')
    w_plt.plot(time,w_z,label=r'$\omega_z$')
    w_plt.set_xlim(0,6000)
    w_plt.set_ylim(-.0023,.002)
    w_plt.set_yticks([-2e-3, 0, 2e-3])
    w_plt.grid(True,alpha=.3)
    w_plt.legend(loc='upper right')
    w_plt.ticklabel_format(axis='y',style='sci', scilimits=(-3,3))
    w_plt.set_title('Body Rates')
    w_plt.set_xlabel('Time(s)')
    w_plt.set_ylabel('Angular Velocity (rad/s)')

    ## quaternion plots
    
    q_plt.plot(time,eps1,label=r'$\epsilon_1$')
    q_plt.plot(time,eps2,label=r'$\epsilon_2$')
    q_plt.plot(time,eps3,label=r'$\epsilon_3$')
    q_plt.plot(time,eta,label=r'$\eta$')

    q_plt.set_xlim(0,6000)
    q_plt.set_ylim(-1,1)
    q_plt.set_yticks([-1,0,1])

    q_plt.set_xlabel('Time(s)')
    q_plt.set_ylabel('Quaternion Paramter')
    q_plt.grid(True,alpha=.3)
    q_plt.legend(loc='upper right')
    q_plt.set_title('Quaternions')

    ###Euler plots###
    e_plt.plot(time,e1,label=r'$\phi$')
    e_plt.plot(time,e2,label=r'$\theta$')
    e_plt.plot(time,e3,label=r'$\psi$')

    e_plt.set_xlim(0,6000)
    e_plt.set_ylim(-500,1000)
    e_plt.set_yticks([-500,0,500,1000])

    e_plt.set_xlabel('Time(s)')
    e_plt.set_ylabel('Angle(deg)')
    e_plt.grid(True,alpha=.3)
    e_plt.legend(loc='upper right')
    e_plt.set_title('Euler Angles')


    plt.tight_layout()
    #plt.show()

    ###Part3###
    sun_ECI = np.array([[1],[0],[0]]) #x side facing to spacecraft

    # Areas and normals
    a_bus = 4*ones((6,1))
    n_bus = np.array([[1, 0, 0],[-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0 ,1], [0, 0, -1]])
    cp_bus = np.array([[1, 0, 0],[-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0 ,1], [0, 0, -1]])
    
    a_p1 = 6*ones((2,1))
    n_p1 = np.array([[0, 0, 1], [0, 0, -1]])  
    cp_p1 = np.array([[0, .25, .025],[0, 2.5,-.025]])
    
    a_p2 = a_p1
    n_p2 = n_p1
    cp_p2 = np.array([[0, -.25, .025],[0, -2.5,-.025]])


    a_sen = np.array([.25, .25, .25, .25, .0625]).reshape((5,1))
    n_sen = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1]])
    cp_sen = np.array([[.25,0,1.5],[-.25,0,1.5],[0, .25, 1.5], [0,-.25,1.5],[0,0,2]]) 
    

    # com_total
    areas_sc = np.vstack((a_bus,a_p1,a_p2,a_sen))
    normals_sc = np.vstack((n_bus,n_p1,n_p2,n_sen))
    cop_sc = np.vstack((cp_bus,cp_p1,cp_p2,cp_sen))


    ###Aero drag###
    '''density model used https://juliaspace.github.io/SatelliteToolboxAtmosphericModels.jl/stable/man/exponential/'''
    C_d = 2.2 # can compare with 2.5 since also acceptable
    R_e = 6378 # [km] 
    rho_0 = 3.614e-14 # 
    H = 109.9
    h_0 = 700 # [km]
    h_rho = norm(r_eci) - R_e
    rho = rho_0*exp(-(h_rho-h_0)/H)
        for i in range(len(time)):
            v_b = quat2C21(q_hist[i])@v_eci
            s_wet = normals_sc*v_b
            if s_wet > 0:
               a_wet = areas_sc*s_wet
               v_b_hat = v_b/norm(v_b) # unit vector for v_b 
               Faero = -1/2*rho*(norm(v_eci)*1000)**2*C_d*a_wet*v_b_hat
               T_aero = cross(cop_sc,Faero.ravel()).reshape(1,-1)
             Taero_tot = np.sum(Tsrp,axis=0)
        Taero_history.append(Taero_tot.copy())
    Taero_hist = np.array(Taero_history)

    
    ###SRP###
    p_sun = 4.5e-6 # [N/m^2] srp from sun 
    Tsrp_history = []
    
    for i in range(len(time)):
        sun_b = quat2C21(q_hist[i])@sun_ECI
        Tsrp = []
        for j in range(len(areas_sc)):
            s_wet = (normals_sc[j].reshape(1,-1)@sun_b)/norm(sun_b)
            if  s_wet > 0:
                a_wet = areas_sc[j]*s_wet
                Fs = p_sun*sun_b*a_wet
                Tsrp_all = cross(cop_sc[j],Fs.ravel()).reshape(1,-1) # cross needs colunms to be blank to work or 2 or 3 elements hence the ravel   
            
                Tsrp.append(Tsrp_all.ravel())
        Tsrp_tot = np.sum(Tsrp,axis=0)
        Tsrp_history.append(Tsrp_tot.copy())
    Tsrp_hist = np.array(Tsrp_history)
    print(Tsrp_hist.shape)

    T_x = (Tsrp_hist[:,0]).reshape(-1,1)
    T_y = (Tsrp_hist[:,1]).reshape(-1,1)
    T_z = (Tsrp_hist[:,2]).reshape(-1,1)

    ###MMG###
    '''Reference ; https://www.mdpi.com/2226-4310/11/11/956'''




    fig , (Tsrp_plt, p_plt, c_plt) = plt.subplots(3,1)

    Tsrp_plt.plot(time,T_x,label=r'$T_x$')
    Tsrp_plt.plot(time,T_y,label=r'$T_y$')
    Tsrp_plt.plot(time,T_z,label=r'$T_z$')

    '''Tsrp_plt.set_xlim(0,6000)
    Tsrp_plt.set_ylim(-500,1000)
    Tsrp_plt.set_yticks([-500,0,500,1000])'''

    Tsrp_plt.set_xlabel('Time(s)')
    Tsrp_plt.set_ylabel('Angle(deg)')
    Tsrp_plt.grid(True,alpha=.3)
    Tsrp_plt.legend(loc='upper right')
    Tsrp_plt.set_title('Tsrp')
    plt.close(1)
    #plt.show()
    
    




    

    
    

    



    













   









    



    





    



     
if __name__ == "__main__":
    main()







