import logging
import unittest
from unittest import TestCase

import numpy as np
import jax.numpy as jnp

import PtyLabX


logging.basicConfig(level=logging.DEBUG)


class TestBaseEngine(TestCase):
    def setUp(self) -> None:
        (
            experimentalData,
            reconstruction,
            params,
            monitor,
            ePIE_engine,
        ) = PtyLabX.easyInitialize("example:simulation_cpm", operationMode="CPM")

        self.reconstruction = reconstruction
        self.ePIE_engine = ePIE_engine

    def test_position_correction(self):
        import time

        rowShifts = np.array(
            [
                -2, -2, -2, -2, -2,
                -1, -1, -1, -1, -1,
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
            ]
        )
        colShifts = np.array([-2, -1, 0, 1, 2] * 5)

        Opatch = jnp.array(np.random.rand(513, 513))
        O = jnp.roll(Opatch, axis=(-2, -1), shift=(1, -1))

        t0 = time.time()
        for i in range(100):
            cc = jnp.zeros((len(rowShifts), 1))
            for shifts in range(len(rowShifts)):
                tempShift = jnp.roll(Opatch, rowShifts[shifts], axis=-2)
                shiftedImages = jnp.roll(tempShift, colShifts[shifts], axis=-1)
                cc = cc.at[shifts].set(jnp.squeeze(jnp.sum(shiftedImages.conj() * O, axis=(-2, -1))))
                del tempShift, shiftedImages
            cc = abs(cc)
            cc = np.asarray(cc.reshape(5, 5))
        t1 = time.time()
        print("CC: ", t1 - t0)

        # new code
        t0 = time.time()
        for i in range(100):
            rowShifts_g, colShifts_g = np.mgrid[-2:3, -2:3]
            rowShifts_g = rowShifts_g.flatten()
            colShifts_g = colShifts_g.flatten()
            FT_O = jnp.fft.fft2(O)
            FT_Op = jnp.fft.fft2(Opatch)
            xcor = jnp.fft.ifft2(FT_O * FT_Op.conj())
            xcor = abs(jnp.fft.fftshift(xcor))
            dy, dx = jnp.unravel_index(jnp.argmax(xcor), xcor.shape)
            dx = int(dx)
        t1 = time.time()
        print("FT: ", t1 - t0)
        N = xcor.shape[-1]
        sy = slice(N // 2 - len(cc) // 2, N // 2 - len(cc) // 2 + len(cc))
        print(" Xcor:")
        print(xcor[sy, sy])

        jnp.allclose(xcor[sy, sy], jnp.array(cc))
