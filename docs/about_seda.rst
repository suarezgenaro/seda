About SEDA
==========

Attribution
-----------
**The SEDA release paper is** `Suárez et al. (2024, in prep.) <https:xxx>`__, but the code was used in `Suárez et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...920...99S/abstract>`__. Please cite the release paper if :math:`\texttt{SEDA}` has contributed to your research. Also make sure to give credits to the models (see :ref:`models`) and other relevant python packages (e.g., see :ref:`seda_overview`) you use via :math:`\texttt{SEDA}`.

Contributing
------------
The :math:`\texttt{SEDA}` package is under active development. Help us improve :math:`\texttt{SEDA}` by reporting `issues <https://github.com/suarezgenaro/seda/issues>`__ on the GitHub repository.

Questions and feedback
----------------------
:math:`\texttt{SEDA}` was developed and is maintained by Genaro Suárez (gsuarez@amnh.org, gsuarez2405@gmail.com). Please reach out with any suggestions, questions, and/or feedback.

Logo
----
Seda is a Spanish word that means silk, which motivates the :math:`\texttt{SEDA}` logo. The logo shows a silk cocoon (irregular grid the code can handle) wrapping key molecules (water, methane, ammonia, and silicates) in the atmospheres of brown dwarfs and gas giant (exo)planets (background image).

FAQs
----
**3. Error opening the generated pickle files: ModuleNotFoundError: No module named 'numpy._core' when openning pickle file.

It typically indicates an issue with the installation of the numpy package. Potential solutions are:

- Reinstall Numpy:

pip uninstall numpy
pip install numpy

- Update Packages
pip uninstall numpy
pip install numpy

**2. Why after cloning SEDA to get an updated version my notebook still reads the old version?**

After cloning the repository, install it (follow the installation steps `here <https://seda.readthedocs.io/en/latest/installation.html>`__. Then restart your notebook and make sure it was opened on the seda environment.

**1. Is there a way to run the code faster, specially the convolution of model spectra?**

The convolution of high-resolution model spectra indeed takes up most of the runtime. You can constrain the ranges of the parameters in the models to convolve only a grid subset with relevant model spectra for your target (see :meth:`~seda.input_parameters.ModelOptions`). As suggested in this `issue <https://github.com/suarezgenaro/seda/issues/14>`__, you can save the convolved model spectra to reuse them and avoid the convolution step to expedite the forward modeling of additional data with a similar resolution.
