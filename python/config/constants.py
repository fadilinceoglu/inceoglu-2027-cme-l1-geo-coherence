"""Physical constants and the provider's UTC reference epoch.

These are fixed conversion constants, separate from study method settings and
measurements derived from the supplied observations.
"""

import pandas as pd


PROTON_MASS_KG = 1.67262192369e-27
BOLTZMANN_J_K = 1.380649e-23
EPOCH_1958 = pd.Timestamp("1958-01-01", tz="UTC")
