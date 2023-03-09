# coverage_with_recharging
the program visualizes the flight of the UAV and the movement of the mobile charging station at the landing points of the quadcopter.
Algorithm:
1. generating a polygon by a given number of vertices
2. Sampling of the terrain area into cells corresponding to the size of the UAV field of view
3. Translation of sampling cells into an undirected terrain graph
4. Application of the wave algorithm of terrain coverage
5. Calculation of the critical charge points of the UAV battery. 
6. Formation of the trajectory of the mobile charging station according to these points at a certain speed
7. The speed is calculated so as to ensure the simultaneous arrival of the UAV and the ground station
8. Visualization of movement with plotting of speed changes via pyplot

Download all files and run main file
