# CMB-2D-grav_lensing-with-nifty
This code does numerical simulations of gravitational lensing of the cosmic microwave background within 2D Cartesian space in Python. Faster than using spherical harmonics, and could be used for the flat-sky approximation.
Requires the numerical information field theory (NIFTY) package.

Remember to change the input/output parameters in FlatLensingProblem, in both the config and Testconfig classes. Rest of document should be fairly well documented. Sample input data provided, but feel free to try the test case.

Many thanks to my supervisors Stefan Hilbert and Torsten Enßlin. And of course, this would not have been possible without much imput and guidance from Vanessa Boehm, Maksim Greiner, Marco Selig, and all other members of the research group.

Writen in the summer of 2014 at the Max Planck Institute of Astrophysics, Garching bei Munchen.

TODO: Wiener Filter code should get an overview.

EXAMPLE (with some more unreasonable dimensions to demonstrate a more obvious effect):

![alt tag](https://raw.github.com/jpbreuer/CMB-2D-grav_lensing-with-nifty/master/output/signal.png)

![alt tag](https://raw.github.com/jpbreuer/CMB-2D-grav_lensing-with-nifty/master/output/Phi.png)

![alt tag](https://raw.github.com/jpbreuer/CMB-2D-grav_lensing-with-nifty/master/output/data.png)


TEST EXAMPLE (test in x direction):

![alt tag](https://raw.github.com/jpbreuer/CMB-2D-grav_lensing-with-nifty/master/testoutput/signal.png)

![alt tag](https://raw.github.com/jpbreuer/CMB-2D-grav_lensing-with-nifty/master/testoutput/Phi.png)

![alt tag](https://raw.github.com/jpbreuer/CMB-2D-grav_lensing-with-nifty/master/testoutput/result.png)