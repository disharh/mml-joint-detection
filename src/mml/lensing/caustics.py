from lenstronomy.LensModel.Solver.epl_shear_solver import caustics_epl_shear

def get_caustics(kwargs_lens, number_images):
        """
        Return the source-plane caustic boundary for a given GW image multiplicity
        (2, 3, or 4) in an EPL+shear lens model.

        Parameters
        ----------
        kwargs_lens : list of dict
            Lens parameters in lenstronomy format for ['EPL', 'SHEAR'].
        number_images : int
            Desired image multiplicity:
            4 - quad (diamond) caustic,
            3 - inner diamond caustic,
            2 - double-image boundary (with finite magnification cutoff).

        Returns
        -------
        ndarray, shape (2, N) [default N=500 is the number of angular sampling points used to trace the caustic curve]
            Caustic coordinates (x, y) in the source plane.
        """
        maginf_cut = -1/0.1 # This shouldn't really be too bad.

        if number_images == 4:
            caustics = caustics_epl_shear(kwargs_lens, return_which='quad')
        elif number_images == 3:
            # Smallest magnification in the 3-image region outside of the diamond within the cut is very small for the central image <0.01, so these are neglected.
            caustics = caustics_epl_shear(kwargs_lens, return_which='caustic')
        elif number_images == 2:
            caustics = caustics_epl_shear(kwargs_lens, return_which='double',
                                           maginf=maginf_cut)  # Don't sample below mu=0.1. Sorta arbtirary limit. There is really no better way to do it I think.
        else:
            raise ValueError("Unsupported number of detected GWs")
        return caustics