"""
Name: SampleEvent
breif: Samples events for particles provided in a phase space for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""
import numpy as np


def SampleEvent(p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index, capture_event_index, fission_event_index, num_part, nu_new_neutrons, rands):
    fissions_to_add = 0
    scat_count = 0
    cap_count = 0
    fis_count = 0
    killed = 0
    
    #normalize cross sections in each mesh cell
    total_scat_xsec = mesh_scat_xsec + mesh_cap_xsec + mesh_fis_xsec
    mesh_scat_xsec /= total_scat_xsec
    mesh_cap_xsec /= total_scat_xsec
    mesh_fis_xsec /= total_scat_xsec
    
    
    for i in range(num_part):
        if p_alive[i] == True:
            
            event_rand = rands[i]
            
            #produce a vector defining the boundaries of the function in a given 
            #pdf_bounds = np.array([0,
            #               mesh_scat_xsec[p_mesh_cell[i]],
            #               mesh_scat_xsec[p_mesh_cell[i]]+mesh_cap_xsec[p_mesh_cell[i]],
            #               mesh_scat_xsec[p_mesh_cell[i]] + mesh_cap_xsec[p_mesh_cell[i]] + mesh_fis_xsec[p_mesh_cell[i]],
            #               1])
            
            #scatter?
            if event_rand < mesh_scat_xsec[p_mesh_cell[i]]:
                scatter_event_index[scat_count] = i
                scat_count += 1
            
            #capture?
            elif mesh_scat_xsec[p_mesh_cell[i]] < event_rand < mesh_scat_xsec[p_mesh_cell[i]]+mesh_cap_xsec[p_mesh_cell[i]]:
                p_alive[i] = False
                killed += 1
                capture_event_index[cap_count] = i
                cap_count +=1
                
            #fission?
            elif mesh_scat_xsec[p_mesh_cell[i]] + mesh_cap_xsec[p_mesh_cell[i]] < event_rand < mesh_scat_xsec[p_mesh_cell[i]] + mesh_cap_xsec[p_mesh_cell[i]] + mesh_fis_xsec[p_mesh_cell[i]]:
                p_alive[i] = False
                killed += 1
                fissions_to_add += nu_new_neutrons
                fission_event_index[fis_count] = i
                fis_count +=1
                
            else:
                print("error: event for particle {part} not sorted properly".format(part =i))
                print("random number used: {rand}".format(rand = event_rand))
                print(pdf_bounds)
                
                exit()
                
    return(scatter_event_index, scat_count, capture_event_index, cap_count, fission_event_index, fis_count)
    
    
def test_SampleEvent():
        p_mesh_cell = np.array([0,1,0,5])
        p_alive = [True,True,True,False]
        
        mesh_cap_xsec = 1/3*np.ones(2)
        mesh_scat_xsec = 1/3*np.ones(2)
        mesh_fis_xsec = 1/2*np.ones(2)
        
        scatter_event_index = np.zeros(3)
        capture_event_index = np.zeros(3)
        fission_event_index = np.zeros(3) 
        
        controled_rands = np.array([.2, .4, .8])
        
        nu = 2
        num_part = 3
        
        [scatter_event_index, scat_count, capture_event_index, cap_count, fission_event_index, fis_count] = SampleEvent(p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index, capture_event_index, fission_event_index, num_part, nu, controled_rands)
        
        assert (fis_count == 1)
        assert (scat_count == 1)
        assert (cap_count == 1)
        
        
        assert (capture_event_index[0] == 1)
        assert (fission_event_index[0] == 2)
        assert (scatter_event_index[0] == 0)
        
if __name__ == '__main__':
    test_SampleEvent()
