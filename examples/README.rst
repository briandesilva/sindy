PySINDy Examples
================

This directory showcases examples of PySINDy in action.  Not all examples are run on
the current master branch. They serve to show what is possible with pysindy, but do
not necessarily use the current API.
Each is copied from another repository that contains dependency information and
potentially a greater description.

Some notebooks require substantial computing resources.

Feature overview
-----------------------------------------------------------------------------------------------------------
This notebook gives an almost exhaustive overview of the different features available in PySINDy. It's a good reference for how to set various options and work with different types of datasets.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./1_feature_overview/example


`Introduction to SINDy <./2_introduction_to_sindy/example.ipynb>`_
---------------------------------------------------------------------------------------------------------------------
We recommend that people new to SINDy start here. We give a gentle introduction to the SINDy method and how different steps in the algorithm are represented in PySINDy. We also show how to use PySINDy to learn a model for a simple linear differential equation.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./2_introduction_to_sindy/example


.. pysindy-example::
    :key: original
    :title: Original Paper

    This repository recreates the results from the `original SINDy paper <https://www.pnas.org/content/pnas/113/15/3932.full.pdf>`_.
    It applies SINDy to the following problems:
    * Linear 2D ODE
    * Cubic 2D ODE
    * Linear 3D ODE
    * Lorenz system
    * Fluid wake behind a cylinder
    * Logistic map
    * Hopf system

`Scikit-learn compatibility <./4_scikit_learn_compatibility/example.ipynb>`_
-------------------------------------------------------------------------------------------------------------------------------
Shows how PySINDy interfaces with various Scikit-learn objects.

* Cross-validation
* Sparse regressors

.. toctree::
    :hidden:
    :maxdepth: 1

    ./4_scikit_learn_compatibility/example

`Differentiation <./5_differentiation/example.ipynb>`_
---------------------------------------------------------------------------------------------------------
Explore the differentiation methods available in PySINDy on pure differentiation problems and as components in the SINDy algorithm.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./5_differentiation/example


`Deeptime compatibility <./6_deeptime_compatibility/example.ipynb>`_
------------------------------------------------------------------------------------------------------------------------
See a demonstration of PySINDy objects designed to conform to the `Deeptime <https://deeptime-ml.github.io/latest/index.html>`_ API.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./6_deeptime_compatibility/example

`Plasma physics <./7_plasma_examples/example.ipynb>`_
----------------------------------------------------------------------------------------------
Use the ``ConstrainedSR3`` optimizer to build a constrained model for the temporal POD modes of a plasma simulation.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./7_plasma_examples/example

`Trapping SINDy <./8_trapping_sindy_examples/example.ipynb>`_
-----------------------------------------------------------------------------------------------------------
This notebook applies the ``TrappingSR3`` optimizer to various canonical fluid systems., proposed in this paper: Kaptanoglu, Alan A., et al. "Promoting global stability in data-driven models of quadratic nonlinear dynamics." Physical Review Fluids 6.9 (2021): 094401. A preprint is found here `<https://arxiv.org/abs/2105.01843>`_.
See as well the other examples in this folder, which explore the extension to local trapping and enstrophy in Peng, Mai et al. "Local stability guarantees for data-driven quadratically nonlinear models."

.. toctree::
    :hidden:
    :maxdepth: 1

    ./8_trapping_sindy_examples/example

`SINDyPI <./9_sindypi_with_sympy/example.ipynb>`_
----------------------------------------------------------------------------------------------
This notebook applies the ``SINDyPI`` optimizer to a simple implicit ODE and was originally proposed in this paper: Kaheman, Kadierdan, J. Nathan Kutz, and Steven L. Brunton. "SINDy-PI: a robust algorithm for parallel implicit sparse identification of nonlinear dynamics." Proceedings of the Royal Society A 476.2242 (2020): 20200279.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./9_sindypi_with_sympy/example

`PDEFIND <./10_PDEFIND_examples/example.ipynb>`_
----------------------------------------------------------------------------------------------
This notebook applies the PDEFIND algorithm (SINDy for PDE identification) to a number of PDEs, and was originally proposed in this paper: Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations." Science Advances 3.4 (2017): e1602614.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./10_PDEFIND_examples/example

`Greedy Algorithms <./11_SSR_FROLS_examples/example.ipynb>`_
-----------------------------------------------------------------------------------------------------
This notebook uses the step-wise sparse regression (SSR) and forward-regression orthogonal least-squares (FROLS) algorithms, which are greedy algorithms that iteratively truncate (or add) one nonzero coefficient at each algorithm iteration.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./11_SSR_FROLS_examples/example

`Weak formulation SINDy <./12_weakform_SINDy_examples/example.ipynb>`_
--------------------------------------------------------------------------------------------------------------
This notebook uses SINDy to identify the weak-formulation of a system of ODEs or PDEs, adding significant robustness against noise in the data.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./12_weakform_SINDy_examples/example

`Model ensembles <./13_ensembling/example.ipynb>`_
----------------------------------------------------------------------------------------------
This notebook uses sub-sampling of the data and sub-sampling of the SINDy library to generate many models, and the user can choose how to average or otherwise combine these models together. This tends to make SINDy more robust against noisy data.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./13_ensembling/example

`Cavity flow <./14_cavity_flow/example.ipynb>`_
----------------------------------------------------------------------------------------------
Demonstrates the use of SINDy to learn a model for the quasiperiodic dynamics in a shear-driven cavity at Re=7500, following Callaham, Brunton, and Loiseau (2021), preprint available here `<https://arxiv.org/pdf/2106.02409>`_.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./14_cavity_flow/example

`PySINDy lectures <./15_pysindy_lectures/example.ipynb>`_
----------------------------------------------------------------------------------------------
Lectures on PySINDy usage from Alan Kaptanoglu's Youtube channel: https://www.youtube.com/playlist?list=PLN90bHJU-JLoOfEk0KyBs2qLTV7OkMZ25.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./15_pysindy_lectures/example

`Noise robustness benchmarking <./16_noise_robustness/example.ipynb>`_
----------------------------------------------------------------------------------------------
Benchmarking PySINDy on low dimensional chaotic systems subject to noise, following Kaptanoglu, Zhang, Nicolaou, Fasel, & Brunton, (2023), preprint available here https://arxiv.org/abs/2302.10787.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./16_noise_robustness/example

`Parameterized pattern formation <./17_parameterized_pattern_formation/example.ipynb>`_
----------------------------------------------------------------------------------------------
Demonstrates the use of SINDyCP for discovery of parameterized pattern formation, following Nicolaou, Huo, Chen, Kutz, and Bruton (2023), preprint available here https://arxiv.org/abs/2301.02673.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./17_parameterized_pattern_formation/example

`All examples including examples under development`_
----------------------------------------------------------------------------------------------

.. toctree::
    :glob:
    :maxdepth: 1

    **
