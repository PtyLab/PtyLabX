import logging
import numpy as np
import jax.numpy as jnp


class LinearProbe:
    def __init__(self):
        self.logger = logging.getLogger("SHG")
        self.probe = None
        self.probe_temp = None

    def clear(self):
        pass

    def push(self, new_probe, index, N_positions, factor=1.0, force=False):
        """
        Set the current estimate of the probe to new_probe.
        """
        if force:
            self.probe = new_probe
        elif self.probe is not None:
            self.probe = new_probe * factor + (1 - factor) * self.probe
        else:
            self.probe = new_probe
        self.probe_temp = self.probe.copy()

    def set_temporary(self, probe):
        """These map to self.reconstruction.probe. Can be used for quick updates in the calculation of the probe.

        Once you're done, make it official by updating with push()"""
        self.probe_temp = probe

    def get_temporary(self):
        return self.probe_temp

    def get(self, index):
        return self.probe

    def roll(self, dy, dx):
        assert self.probe_temp is not None
        self.probe = self.probe_temp.copy()
        self.probe = jnp.roll(self.probe, (-dy, -dx), axis=(-2, -1))
        self.probe_temp = self.probe.copy()


class SHGProbe(LinearProbe):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("SHG")
        self.probe = None  # wavelength = wavelength  * self.nonlinearity
        self.nonlinearity = 2

    def clear(self):
        pass

    def push(self, new_probe_nonlinear, index, N_positions):  # ty: ignore[invalid-method-override]
        """Gets the update of the nonlinear part"""
        if self.probe is not None:
            new_probe_nonlinear = jnp.array(new_probe_nonlinear)
        # what's actually pushed is the second harmonic. We need to update that

        if self.probe is None:
            self.probe = new_probe_nonlinear * 0
        # try "newtons" method

        # Solve for the new estimate, and update the original estimate based on it
        new_probe_estimate = new_probe_nonlinear ** (1.0 / self.nonlinearity)

        diff = new_probe_estimate - self.probe
        self.probe = self.probe + diff / (2 * self.nonlinearity)
        if N_positions == -1:
            print(np.linalg.norm(np.asarray(self.probe**self.nonlinearity - new_probe_nonlinear)))
        self.probe_temp = self.probe.copy() ** self.nonlinearity

    def change_nonlinearity(self, nonlinearity):
        last_probe = self.get(None).copy()
        self.nonlinearity = nonlinearity
        self._push_hard(last_probe)

    def _push_hard(self, new_probe, number_of_iterations=50):
        new_probe = jnp.array(new_probe)
        for i in range(number_of_iterations):
            self.push(new_probe, None, -1)
            print(jnp.linalg.norm(self.get(None) - new_probe))

    def get(self, index):
        assert self.probe is not None
        assert self.nonlinearity is not None
        return self.probe**self.nonlinearity

    def get_fundamental(self):
        return self.probe
