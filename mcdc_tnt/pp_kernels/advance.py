"""
Name: Advance
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import math
import numpy as np


def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L):
    """
    Guts of transport is the function that actaully moves particles around, go figure.
    Implements surface tracking with flux (w/ error) via track length estimator

    Parameters
    ----------
    p_pos_x : vector double
        PSV: x position of phase space particles (index is particle value).
    p_pos_y : vector double
        PSV: y position of phase space particles (index is particle value).
    p_pos_z : vector double
        PSV: z position of phase space particles (index is particle value).
    p_mesh_cell : vector int
        PSV: mesh cell location of a given particle.
    dx : double
        mesh cell width.
    p_dir_y : vector double
        PSV: y direction unit value of phase space particles (index is particle value).
    p_dir_z : vector double
         PSV: z direction unit value of phase space particles (index is particle value).
    p_dir_x : vector double
         PSV: x direction unit value of phase space particles (index is particle value).
    p_speed : vector double
        PSV: speed (energy) or a particle (index is particle).
    p_time : vector double
        PSV: particle clock.
    num_part : int
        number of particles currently under transport.
    mesh_total_xsec : vector double
        total cross section of every mesh cell (length num_cells).
    mesh_dist_traveled : vector double
        track length estimator tally for use in comp of flux.
    mesh_dist_traveled_squared : TYPE
        distance a particle travels in each cell for use in error with flux.
    L : double
        length of slab.

    Returns
    -------
    Updated PSV with mesh distances.

    """
    p_end_trans = np.zeros(num_part)
    end_flag = 1
    max_mesh_index = len(mesh_total_xsec)-1
    
    while end_flag == 1:
        #allocate randoms
        rands = np.random.random([num_part])
        #vector of indicies for particle transport
        
        p_dist_travled = np.zeros(num_part)
        
        pre_p_mesh = p_mesh_cell
        
        for i in range(num_part):
            Advance_cycle(i, p_pos_x, p_pos_y, p_pos_z,
                          p_dir_y, p_dir_z, p_dir_x, 
                          p_mesh_cell, p_speed, p_time,  
                          dx, mesh_total_xsec, L,
                          p_dist_travled, p_end_trans, rands)
        #accumulate mesh distance tallies (pk.for tallies
        end_flag = 0
        for i in range(num_part):
            if (0 < pre_p_mesh[i] < max_mesh_index):
                mesh_dist_traveled[pre_p_mesh[i]] += p_dist_travled[i]
                mesh_dist_traveled_squared[pre_p_mesh[i]] += p_dist_travled[i]**2
                
            if p_end_trans[i] == 1:
                end_flag = 1
    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared)




def Advance_cycle(i, p_pos_x, p_pos_y, p_pos_z,
                  p_dir_y, p_dir_z, p_dir_x, 
                  p_mesh_cell, p_speed, p_time,  
                  dx, mesh_total_xsec, L,
                  p_dist_travled, p_end_trans, rands):

    kicker = 1e-10

    if (p_end_trans[i] == 0):
        if (p_pos_x[i] < 0): #exited rhs
            p_end_trans[i] = 0
        elif (p_pos_x[i] >= L): #exited lhs
            p_end_trans[i] = 0
            
        else:
            dist = -math.log(rands[i]) / mesh_total_xsec[p_mesh_cell[i]]
            
            x_loc = (p_dir_x[i] * dist) + p_pos_x[i]
            LB = p_mesh_cell[i] * dx
            RB = LB + dx
            
            if (x_loc < LB):        #move partilce into cell at left
                p_dist_travled[i] = (LB - p_pos_x[i])/p_dir_x[i] + kicker
                cell_next = p_mesh_cell[i] - 1
               
            elif (x_loc > RB):      #move particle into cell at right
                p_dist_travled[i] = (RB - p_pos_x[i])/p_dir_x[i] + kicker
                cell_next = p_mesh_cell[i] + 1
                
            else:                   #move particle in cell
                p_dist_travled[i] = dist
                p_end_trans[i] = 0
                cell_next = p_mesh_cell[i]
                
            p_pos_x[i] += p_dir_x[i]*p_dist_travled[i]
            p_pos_y[i] += p_dir_y[i]*p_dist_travled[i]
            p_pos_z[i] += p_dir_z[i]*p_dist_travled[i]
            
            p_mesh_cell[i] = cell_next
            p_time[i]  += p_dist_travled[i]/p_speed[i]





def StillIn(p_pos_x, surface_distances, p_alive, num_part):
    tally_left = 0
    tally_right = 0
    for i in range(num_part):
        #exit at left
        if p_pos_x[i] <= surface_distances[0]:
            tally_left += 1
            p_alive[i] = False
            
        elif p_pos_x[i] >= surface_distances[len(surface_distances)-1]:
            tally_right += 1
            p_alive[i] = False
            
    return(p_alive, tally_left, tally_right)
    
    
    
    
def test_Advance():
    L = 1
    dx = .25
    N_m = 4
    
    num_part = 6
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1])
    p_pos_y = 2.1*np.ones(num_part)
    p_pos_z = 3.4*np.ones(num_part)
    
    p_mesh_cell = np.array([-1, 0, 0, 1, 3, 4])
    
    p_dir_x = np.ones(num_part)
    p_dir_x[0] = -1
    p_dir_y = np.zeros(num_part)
    p_dir_z = np.zeros(num_part)
    
    p_speed = np.ones(num_part)
    p_time = np.zeros(num_part)
    p_alive = np.ones(num_part, bool)
    p_alive[5] = False
    
    
    particle_speed = 1
    mesh_total_xsec = np.array([0.1,1,.1,100])
    
    mesh_dist_traveled_squared = np.zeros(N_m)
    mesh_dist_traveled = np.zeros(N_m)
    
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared] = Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L)
    
    
    assert (np.sum(mesh_dist_traveled) > 0)
    assert (np.sum(mesh_dist_traveled_squared) > 0)
    assert (p_pos_x[0]  == -.01)
    assert (p_pos_x[5]  == 1.1)
    assert (p_pos_x[1:4].all()  > .75)
    
    
        
def test_StillIn():    
    
    num_part = 7
    surface_distances = [0,.25,.75,1]
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1, 1])
    p_alive = np.ones(num_part, bool)
    
    [p_alive, tally_left, tally_right] = StillIn(p_pos_x, surface_distances, p_alive, num_part)
    
    assert(p_alive[0] == False)
    assert(p_alive[5] == False)
    assert(tally_left == 2)
    assert(tally_right == 2)
    assert(p_alive[2:4].all() == True)


if __name__ == '__main__':
    test_Advance()
    test_StillIn()
   

    