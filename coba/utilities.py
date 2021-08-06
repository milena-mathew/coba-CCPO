"""Simple one-off utility methods with no clear home."""

import sys
import os

from io import UnsupportedOperation
from contextlib import contextmanager
from typing import IO

import numpy as np
from numpy.core.records import array

@contextmanager
def redirect_stderr(to: IO[str]):
    """Redirect stdout for both C and Python.

    Remarks:
        This code comes from https://stackoverflow.com/a/17954769/1066291. Because this modifies
        global pointers this code is not "thread-safe". This limitation is also true of the built-in
        Python modules such as `contextlib.redirect_stdout` and `contextlib.redirect_stderr`. See
        https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout for more info.
    """
    try:
        #we assume that this fd is the same
        #one that is used by our C library
        stderr_fd = sys.stderr.fileno()

        def _redirect_stderr(redirect_stderr_fd):
            
            #first we flush Python's stderr. It should be noted that this
            #doesn't close the file descriptor (i.e., sys.stderr.fileno())
            #or Python's wrapper around the stderr_fd.
            sys.stderr.flush()
        
            # next we change the stderr_fd to point to the
            # file contained in the redirect_stderr_fd.
            # If C has anything buffered for stderr it
            # will now go to the new fd. There do appear
            # to be ways to flush C buffers from Python 
            # but I'm not sure it is worth it given the
            # amount of complexity it adds to the code.
            # This change also means that sys.stderr now
            # points to a new file since sys.stderr points
            # to whatever file is at stderr_fd
            os.dup2(redirect_stderr_fd, stderr_fd)

        # when we dup there are now two fd's
        # pointing to the same file. Closing
        # one of these doesn't close the other.
        # therefore it is on us to close the
        # duplicate fd we make here before ending.
        old_stderr_fd = os.dup(stderr_fd)
        new_stderr_fd = to.fileno()

        try:
            _redirect_stderr(new_stderr_fd)
            yield # allow code to be run with the redirected stderr
        finally:
            _redirect_stderr(old_stderr_fd) 
            os.close(old_stderr_fd)
    except UnsupportedOperation:
        #if for some reason we weren't able to redirect
        #then simply move on. No reason to stop working.
        yield

class PackageChecker:

    @staticmethod
    def matplotlib(caller_name: str) -> None:
        """Raise ImportError with detailed error message if matplotlib is not installed.

        Functionality requiring matplotlib should call this helper and then lazily import.

        Args:    
            caller_name: The name of the caller that requires matplotlib.

        Remarks:
            This pattern borrows heavily from sklearn. As of 6/20/2020 sklearn code could be found
            at https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py
        """
        try:
            import matplotlib # type: ignore
        except ImportError as e:
            raise ImportError(
                caller_name + " requires matplotlib. You can "
                "install matplotlib with `pip install matplotlib`."
            ) from e

    @staticmethod
    def vowpalwabbit(caller_name: str) -> None:
        """Raise ImportError with detailed error message if vowpalwabbit is not installed.

        Functionality requiring vowpalwabbit should call this helper and then lazily import.

        Args:    
            caller_name: The name of the caller that requires matplotlib.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            import vowpalwabbit # type: ignore
        except ImportError as e:
            raise ImportError(
                caller_name + " requires vowpalwabbit. You can "
                "install vowpalwabbit with `pip install vowpalwabbit`."
            ) from e

    @staticmethod
    def pandas(caller_name: str) -> None:
        """Raise ImportError with detailed error message if pandas is not installed.

        Functionality requiring pandas should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires pandas.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            import pandas # type: ignore
        except ImportError as e:
            raise ImportError(
                caller_name + " requires pandas. You can "
                "install pandas with `pip install pandas`."
            ) from e

    @staticmethod
    def numpy(caller_name: str) -> None:
        """Raise ImportError with detailed error message if numpy is not installed.

        Functionality requiring numpy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires numpy.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            import numpy # type: ignore
        except ImportError as e:
            raise ImportError(
                caller_name + " requires numpy. You can "
                "install numpy with `pip install numpy`."
            ) from e

    @staticmethod
    def sklearn(caller_name: str) -> None:
        """Raise ImportError with detailed error message if numpy is not installed.

        Functionality requiring numpy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires numpy.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            import sklearn # type: ignore
        except ImportError as e:
            raise ImportError(
                caller_name + " requires sklearn. You can "
                "install sklearn with `pip install sklearn`."
            ) from e

class DiagonalFreeGrad: 
    """
    Created on Thu Jun 17 16:40:05 2021
    @author: Zakaria Mhammedi
    """
    def __init__(self, d, restart=False, project=False, autoradius=True, radius=1, epsilon=1):
        # Initialize the "sufficient statistics"
        self.d = d
        self.w = np.zeros(d)     # Prediction vector
        self.G = np.zeros(d)     # Sum of gradients 
        self.V = np.zeros(d)     # Sum of squared coordinate-wise gradients
        self.S = np.zeros(d)     # Sum of normalized coordinate-wise gradients (used for restarts)
        self.h1 = np.zeros(d)    # Absolute values of initial non-zero coordinate-wise gradients
        self.ht = np.zeros(d)    # Maximum absolute values of coordinate-wise gradients up to t
        self.Ht = 0
        self.sum_normalized_grad_norm = 1 # Sum of normalized gradients (used for projections)
        
        # Initializing the loss
        self.L = 0

        self.restart = restart
        self.project = project
        self.autoradius = autoradius
        self.radius = radius
        self.epsilon = epsilon

    def update(self, gradientfn):
        # do what was inside the loop, accumulate into state variables
        # Get the feature vector and the label
        
        def norm(x):
            return np.sqrt(np.dot(x,x))
            
        # Norm of prediction 
        w_norm = norm(self.w)

        if self.autoradius:
            project_radius = self.epsilon * np.sqrt(self.sum_normalized_grad_norm)
        else:
            project_radius = self.radius
          
        
        w_project = self.w
        if self.project:
            if w_norm > project_radius:
                print('Projected')
                w_project = self.w * project_radius/w_norm
        
        # Compute gradient information
        g = gradientfn(w_project)
        
        # Cutkosky's varying constrains' reduction:
        # Alg. 1 in http://proceedings.mlr.press/v119/cutkosky20a/cutkosky20a.pdf with sphere sets
        if self.project and w_norm > project_radius and np.dot(g,self.w) < 0:
            tilde_g = g - np.dot(g,self.w/w_norm) * self.w/w_norm 
        else:
            tilde_g = g

        if isinstance(tilde_g, float):
            tilde_g = np.array([tilde_g])
            
        clipped_g = tilde_g
        
        # Update stuff
        for i in range(self.d):
            # Clip the gradient
            abs_tilde_g  = abs(tilde_g[i])
            
            # Only do something if non-zero grad observed
            if abs_tilde_g == 0:
                continue
            
            # Update the hints
            tmp_ht = self.ht[i]
            if self.h1[i]==0:
                self.h1[i]=abs_tilde_g
                self.ht[i]=abs_tilde_g
                self.V[i]+=tilde_g[i]**2
            elif abs_tilde_g > tmp_ht:
                clipped_g[i] *= tmp_ht/abs_tilde_g  
                self.ht[i] = abs_tilde_g
                
            # Check for restarts
            if self.restart and self.ht[i]/self.h1[i] > self.S[i]+2:
                print('Restarted')
                self.h1[i]=self.ht[i]
                self.G[i] = clipped_g[i]
                self.V[i] = clipped_g[i]**2
            else:
                self.G[i] += clipped_g[i]
                self.V[i] += clipped_g[i]**2
                
            # Check this is the same as the implementation (the tmp_ht part)
            if tmp_ht>0:
                self.S[i] += abs(clipped_g[i])/tmp_ht
                
        # Compute prediction
        absG = abs(self.G)
        self.w = - self.G * self.epsilon * self.h1**2 *(2*self.V+self.ht*absG)/(2*(self.V+self.ht*absG)**2 * np.sqrt(self.V))\
            * np.exp(absG**2/(2 * self.V + 2 * self.ht * absG))
       
        # Update statistics for projections
        norm_clipped_g = norm(clipped_g)
        if norm_clipped_g > self.Ht:
            self.Ht = norm_clipped_g
        if self.Ht > 0:
            self.sum_normalized_grad_norm +=  norm_clipped_g/self.Ht 
            
        return self.w