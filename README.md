# quantum-LHO
Simple examples of time evolution with quantum harmonic oscillators.

Please use python 3.4 or higher and install requirements.

    python -m pip install -r requirements.txt
  
In file LHO.py you can shange the shape of initial wavepocket. In this case `wave_gauss()` is defined locally. For LHO gaussian pocket evaluation run 

    python LHO.py --LHO

For coherent states, adjust complex number alpha in script and run

    python LHO.py --coherent
  
If you want to display animation istead of saving movies add `--show` flag

    python LHO.py --LHO --show
