# pyKrig
Kriging & functional analysis of variance (ANOVA) with python

## Installation
(pyKrig requires a C compiler. 
For a Windows + Python3.5/3.6 environment,
[Visual C++ 2015 Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools) is recommended.)

Download the repository and type

```commandline
pip install .
```

or if you have git installed, simply type

```commandline
pip install git+https://github.com/SaTa999/pyPanair
```

## Requirements
pyPanair requires python 3.5+  
(tests have only been performed for python 3.6)  
An [Anaconda3](https://www.continuum.io/) environment is recommended.

## References
Check the following books / papers to learn more about what is implemented in pyKrig
- Jones, D.R., Schonlau, M. and Welch, W.J., "Efficient Global Optimization of Expensive Black-Box Function,"
 *Journal of Global Optimization*, Vol. 13, 1998, pp. 455-492.
- Forrester, A. I. J., and Keane, A. J.; Bressloff, N. W., "Design and Analysis of “Noisy” Computer Experiments,"
 *AIAA Journal*, Vol. 44, No. 10, 2006, pp. 2331-2339.
- Forrester, A. I. J., Sobester, A., and Keane, A. J. *Engineering Design via Surrogate Modelling: A Practical Guide,*
 John Wiley & Sons, 2008.
- Toal, D. J. J., Bressloff, N. W., Keane, A. J., and Holden, C. M. E., "The Development of a Hybridized Particle Swarm
 for Kriging Hyperparameter Tuning," Engineering Optimization, Vol. 43, No. 6, 2011, pp. 675-699.
- Sun, J., Feng, B., and Xu, W., "Particle Swarm Optimization with Particles Having Quantum Behavior,"
 IEEE Proceedings of Congress on Evolutionary Computation, 2004, pp. 325-331.