import numpy as np
import matplotlib.pylab as plt
import dump

Avogadro = 6.023e23
Boltzmann = 1.38e-23

def wallHitCheck(pos, vels, box):
   
    ndims = len(box)
    
    for i in range(ndims):
        vels[(pos[:,i] <= box[i][0]) | (pos[:,i] >= box[i][1])] *= -1

def leapfrog_position(pos, vels, acc, dt):
    
    pos += vels * dt + 0.5 * acc * dt**2
    return pos

def leapfrog_velocity(vels, acc, acc_new, dt):
   
    vels += 0.5 * (acc + acc_new) * dt
    return vels

def leapfrog(pos, vels, acc, dt, force, mass):
    
    # Update position (half-step velocity)
    pos = leapfrog_position(pos, vels, acc, dt)
    
    # Compute new acceleration
    acc_new = forces / mass[np.newaxis].T
    
    # Update velocity (full step)
    v = leapfrog_velocity(vels, acc, acc_new, dt)
    
    # Update acceleration
    acc = acc_new
    
    return pos, vels, acc    

def integrate(pos, vels, forces, mass, dt):
   
    pos += vels * dt
    vels += forces * dt / mass[np.newaxis].T
    

def computeForce(mass, vels, temp, relax, dt):
    
    natoms, ndims = vels.shape
    
    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T
    
    force = - vels * mass[np.newaxis].T / relax + noise
    
    return force

def run(**args):

    natoms, box, dt, temp = args['natoms'], args['box'], args['dt'], args['temp']
    mass, relax, nsteps = args['mass'], args['relax'], args['steps']
    ofname, freq, radius = args['ofname'], args['freq'], args['radius']
    
    ndims = len(box)
    pos = np.random.rand(natoms, ndims)
    

    for i in range(ndims):
        pos[:,i] = box[i][0] + (box[i][1] - box[i][0]) * pos[:,i]
    
    vels = np.random.rand(natoms, ndims)
    mass = np.ones(natoms)* mass / Avogadro
    radius = np.ones(natoms) * radius
    step = 0
    
    output = []

    while step <= nsteps:

        step += 1

        forces = computeForce(mass, vels, temp, relax, dt)

        integrate(pos, vels, forces, mass, dt)

        wallHitCheck(pos, vels, box)

        inst_temp = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2))/(Boltzmann * 3 * natoms)
        output.append([dt*step, inst_temp])
        
        if not step%freq:
            dump.writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, vels=vels)

    return np.array(output)

if __name__ == '__main__':

    params = {
        'natoms': 90,
        'radius': 53e-12,
        'mass': 1e-3,
        'dt': 1e-15,
        'relax':1e-13,
        'temp': 10,
        'steps': 100000,
        'freq': 100,
        'box': ((0, 3e-8), (0, 3e-8), (0, 3e-8)),
        'ofname': 'traj-hydrogen.dump'
        }
    output = run(**params)

    plt.plot(output[:,0] * 1e12, output[:,1])
    plt.xlabel('Time (ps)')
    plt.ylabel('Temp (K)')
    plt.show()


