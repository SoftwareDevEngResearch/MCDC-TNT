from input_parser import SimulationSetup
#import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse


def run():
    """
    main function to run a single generation and plot the output

    Returns
    -------
    Plots and output tables if requested.

    """
    parser = argparse.ArgumentParser(description='Main file to run MC/DC-TNT')
    parser.add_argument('-i', '--input', required=True,
                        help='input file in a .yaml format (see InputDeck.py)')
    parser.add_argument('-o', '--output', required=False,
                        help='output file, if none then output.txt')
                        
    args = parser.parse_args(sys.argv[1:])

    input_file = args.input
    output_file = args.output

    [comp_parms, sim_perams, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, surface_distances] = SimulationSetup(input_file)
    
    if comp_parms['hard_targ'] == 'pp':
        import generations as generations
        print('>>>Running Prue Python kernels (slow)')
    elif comp_parms['hard_targ'] == 'nb_cpu':
        import generations as generations
        print('>>>Running Numba CPU kernels')
    elif comp_parms['hard_targ'] == 'nb_gpu':
        import generations as generations
        print('>>>Running Numba GPU kernels (slow)')
    elif comp_parms['hard_targ'] == 'pyk_cpu':
        import generations_pyk as generations
        print('>>>Running PyKokkos CPU kernels')
        print('    ensure correct conda enviroment is loaded!')
    elif comp_parms['hard_targ'] == 'pyk_gpu':
        print('>>>Feature not yet implemented, running pyk cpu kerenels')
        import generations_pyk as generations
        print('>>>Running PyKokkos CPU kernels')
        print('    ensure correct conda enviroment is loaded!')
    else:
        print()
        print('>>FATAL ERROR: NO HARDWARE TARGET<<')
        print()
        return()
    print()
    
    [scalar_flux, standard_deviation_flux] = generations.Generations(comp_parms, sim_perams, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, surface_distances)

    if (output_file == None):
        print("Simulation Complete, no outputs")

    x_mesh = np.linspace(0,1,len(scalar_flux))

    scalar_flux /= np.max(scalar_flux)
    
    
    if comp_parms['plot error'] == True:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(x_mesh, standard_deviation_flux, '-k')
        plt.title(["$σ^2$ ",comp_parms['sim name']])
        plt.ylabel("$σ^2$")
        plt.xlabel("x [cm]")
        plt.savefig('error.png', dpi=500, facecolor='w', edgecolor='k',orientation='portrait')
        
    if comp_parms['plot flux'] == True:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.plot(x_mesh, scalar_flux, '-b')
        plt.title(["Scalar Flux: ",comp_parms['sim name']])
        plt.ylabel("$\phi [cm^{-2}s^{-1}]$")
        plt.xlabel("x [cm]")
        plt.savefig('sflux.png', dpi=500, facecolor='w', edgecolor='k',orientation='portrait')

if __name__ == "__main__":
    run()
