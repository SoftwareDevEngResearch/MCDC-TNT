"""
Name: CleanUp
breif: Misc functions for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import numpy as np
import pykokkos as pyk
#import numba as nb

@pk.workunit
class SourceParticles:
    
    def __init__(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive,
        num_parts, meshwise_fission_pdf, particle_speed, rands):
        
        self.p_pos_x = pk.from_numpy(p_pos_x)
        self.p_pos_y = pk.from_numpy(p_pos_y)
        self.p_pos_z = pk.from_numpy(p_pos_z)
        
        self.p_dir_x = pk.from_numpy(p_dir_x)
        self.p_dir_y = pk.from_numpy(p_dir_y)
        self.p_dir_z = pk.from_numpy(p_dir_z)
        
        self.p_mesh_cell = pk.from_numpy(p_mesh_cell)
        self.p_speed = pk.from_numpy(p_speed)
        self.p_time = pk.from_numpy(p_time)
        
        self.rands = pk.from_numpy(rands)
        
        self.num_part: int = num_part
        self.dx: pk.double = dx
        self.L: pk.double = L
        
    
    @pk.main
    def Source(self)
        pk.parallel_for(self.num_parts, SourcePK)
    
    @pk.callback
    def ReturnSource(self):
        return(self.p_pos_x, self.p_pos_y, self.p_pos_z, self.p_mesh_cell, self.p_dir_y, self.p_dir_z, self.p_dir_x, self.p_speed, self.p_time)
    
    @pk.workunit
    def SourcePK(i: int):
        
        #find mesh cell birth based on provided pdf
        cell: int = 0
        summer: int = 0
        while (summer < self.rands[i*4]):
            summer += self.meshwise_fission_pdf[cell]
            cell += 1
                
        cell -=1
        self.p_mesh_cell[i] = int(cell)
        
        #sample birth location within cell
        self.p_pos_x[i] = dx*cell + dx*self.rands[i*4+1]
        self.p_pos_y[i] = 0.0
        self.p_pos_z[i] = 0.0
        
        
        # Sample polar and azimuthal angles uniformly
        mu: pk.double  = 2.0*self.rands[i*4+2] - 1.0
        azi: pk.double = 2.0*self.rands[i*4+3]
    
        # Convert to Cartesian coordinate
        c = (1.0 - mu**2)**0.5
        self.p_dir_y[i] = math.cos(azi)*c
        self.p_dir_z[i] = math.sin(azi)*c
        self.p_dir_x[i] = mu

        # Speed
        self.p_speed[i] = particle_speed

        # Time
        self.p_time[i] = 0.0
        
        self.p_alive[i] = True



def test_SourceParticles():
    num_parts = 5
    p_pos_x = np.zeros(num_parts)
    p_pos_y = np.zeros(num_parts)
    p_pos_z = np.zeros(num_parts)
    
    p_mesh_cell = np.zeros(num_parts)
    
    p_dir_x = np.zeros(num_parts)
    p_dir_y = np.zeros(num_parts)
    p_dir_z = np.zeros(num_parts)
    
    p_speed = np.zeros(num_parts)
    p_time = np.ones(num_parts)
    p_alive = np.zeros(num_parts)
    
    particle_speed = 1
    meshwise_fission_pdf = [0,1]
    
    iso=False
    
    dx = 0.2
    
    rands = np.random.random(4*num_parts)
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive] = pk.execute(pk.ExecutionSpace.OpenMP, SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_parts, meshwise_fission_pdf, particle_speed, rands))
    
    #[p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive] = SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_parts, meshwise_fission_pdf, particle_speed, rands)
    
    print("Ran")
    
    assert (np.sum(p_time) == 0)
    assert (p_mesh_cell.all() == 1)
    assert (p_alive.all() == True)
    assert (p_pos_x.all() > .2)
    
if __name__ == '__main__':
    test_SourceParticles()    

