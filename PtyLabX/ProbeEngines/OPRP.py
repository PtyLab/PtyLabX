from scipy import ndimage

import logging

import numpy as np
import jax.numpy as jnp


class OPRP_storage:
    def __init__(self, N_probes=5, correct_position=True):
        self.logger = logging.getLogger("OPRP")
        self.N_probes = N_probes
        self.correct_position = correct_position

    def clear(self):
        """Use this when the probe is changed, for instance number of probe modes is changed"""
        for name in ["probes", "probe_indices", "A", "s", "At"]:
            if hasattr(self, name):
                delattr(self, name)

    def push(self, probe, index, N_positions):
        probe = probe * jnp.exp(-1j * jnp.angle(probe.sum()))

        if not hasattr(self, "probes"):
            self._prepare_probes(probe, N_positions)
            # if it's the first one, make it small so the updates are relatively important
            probe_norm = probe.reshape(self.new_probe_shape)
            probe_norm = probe_norm / jnp.linalg.norm(probe_norm)
            self.push(probe_norm.reshape(self.original_probe_shape), index, N_positions)
            return

        if self.correct_position:
            probe = self.center_probe(probe, index)

        self.probes = self.probes.at[index].set(probe.reshape(self.new_probe_shape))
        self.probe_indices = self.probe_indices.at[index].set(True)

    def tsvd(self):
        self.logger.info("Running TSVD")
        if not np.all(np.asarray(self.probe_indices)):
            indices = jnp.argwhere(self.probe_indices)
            probes = self.probes[indices]
        else:
            probes = self.probes
        average_power = jnp.mean(jnp.abs(probes**2))
        probe_power = jnp.mean(jnp.abs(probes**2), axis=-1, keepdims=True)
        probes = probes * jnp.mean(average_power / (probe_power + 1e-6))

        A, s, At = jnp.linalg.svd(probes, full_matrices=False)
        N = self.N_probes
        # calculate effective rank
        pk = s / jnp.linalg.norm(s.flatten(), ord=1)
        H = -jnp.sum(pk * jnp.log(pk))
        eRank = jnp.exp(H)

        self.A = A[:, :N]
        self.s = s[:N]
        self.At = At[:N]

        self.logger.info(f"Effective rank: {eRank}, truncating to {self.N_probes} modes")

        self.logger.info(f"Average displacement: {np.mean(abs(self.center_mass))}")

    def get(self, index):
        """Get the TSVD estimate of the i-th index.

        If the particular index has not been given yet,
        or tsvd has not been run yet, return the averaged probe that we measured so far."""
        if not hasattr(self, "A"):
            # tsvd has not been run yet, return the averaged probe
            return self.probes[self.probe_indices].mean(axis=0).reshape(self.original_probe_shape)

        if self.probe_indices[index]:  # we measured this one, all easy
            A = self.A[index]
        else:  # tsvd has been run, but this particular probe was not given yet
            # beginning, we ask for a probe that we haven't measured yet.
            # In this case, return the typical probe to have some idea
            # Aka set self.A to [1, 0, 0,... 0]
            # we didn't measure it, return the first mode multiplied with the average power
            A = self.A[0].copy()
            A = A.at[1:].set(0)
            A = A.at[0].set(1.0 * jnp.sign(self.A[0, 0]))
        probe = jnp.matmul(A, self.s[..., None] * self.At)
        # move it back
        probe = probe.reshape(self.original_probe_shape)
        if self.correct_position:
            probe = self.uncenter_probe(probe, index)

        return probe

    def center_probe(self, probe, index):
        dpos = np.array(ndimage.center_of_mass(np.asarray(abs(probe**2)))) - np.array(probe.shape) / 2
        dpos = np.clip(dpos, -2.5, 2.5)
        self.center_mass[index] += 0.01 * dpos
        # move it
        for dim, shift in enumerate(self.center_mass[index]):
            if self.original_probe_shape[dim] != 1:
                shift = int(np.round(shift))
                probe = jnp.roll(probe, shift=-shift, axis=dim)
        return probe

    def uncenter_probe(self, probe, index):
        probe = probe.reshape(self.original_probe_shape)
        for dim, shift in enumerate(self.center_mass[index]):
            if self.original_probe_shape[dim] != 1:
                shift = int(np.round(shift))
                probe = jnp.roll(probe, shift=shift, axis=dim)
        return probe

    def _prepare_probes(self, single_probe, N_positions):
        self.original_probe_shape = single_probe.shape
        self.new_probe_shape = np.array([np.prod(np.array(single_probe.shape))])
        self.N_positions = N_positions
        self.probes = jnp.zeros((self.N_positions, *self.new_probe_shape), dtype=jnp.complex64)
        self.probe_indices = jnp.zeros(self.N_positions, dtype=jnp.bool_)

        if self.correct_position:
            shape = (self.N_positions, len(self.original_probe_shape))
            self.center_mass = np.zeros(shape)

    def estimate_CM(self):
        from scipy import ndimage

        for i in range(self.N_positions):
            probe = self.get(i)
            cmass = np.array(ndimage.center_of_mass(np.asarray(abs(probe) ** 2)))

            print(i, cmass - np.array(probe.shape) / 2 + 1)
