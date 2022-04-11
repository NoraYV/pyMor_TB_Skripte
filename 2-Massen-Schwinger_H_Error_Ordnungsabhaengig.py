import statistics
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor
import time

plt.rcParams['axes.grid'] = True

k1 = 10
k2 = 10
d1 = 1
d2 = 1
k = 4
E= np.array([[1,0,0,0], [0,1,0,0], [0, 0, 1, 0], [0, 0, 0, 1]])
A= np.array([[0,0,1,0], [0,0,0,1], [-k1-k2, k2, -d1-d2, -d2], [k2, -k2, -d2, d2]])
B = np.array([[0], [0], [0], [1]])
C = np.array([[1,0,0,0]])

fom = LTIModel.from_matrices(A, B, C, E=E)

#Reductor schaffen
bt = BTReductor(fom)

#Fehlerschranke pro Ordnung
error_bounds = bt.error_bounds()
print('error bounds:', '\n', error_bounds)
hsv = fom.hsv()
fig, ax = plt.subplots()
ax.semilogy(range(1, len(error_bounds)+1), error_bounds, '.-')
ax.semilogy(range(1, len(hsv)), hsv[1:], '.-')
ax.legend(['error_bounds', 'hsv'])
ax.set_xlabel('Reduced order')
_ = ax.set_title(r'Upper and lower $\mathcal{H}_\infty$ error bounds')
plt.show()

#Reduktion auf ein System 3., 2. und 1. Ordnung

rom3 = bt.reduce(3)
rom2 = bt.reduce(2)
rom1 = bt.reduce(1)

#print('FOM:', fom,'\n', 'ROM',  rom)

w = np.logspace(-2, 8, 300)

fig, axs = plt.subplots(2, 1, figsize=(6, 12), sharex=True, constrained_layout=True, squeeze=False)
fom.transfer_function.bode_plot(w, ax=axs)
rom3.transfer_function.bode_plot(w, ax=axs, linestyle='--')
rom2.transfer_function.bode_plot(w, ax=axs, linestyle='--')
rom1.transfer_function.bode_plot(w, ax=axs, linestyle='--')
plt.legend(['fom', 'rom3', 'rom2', 'rom1'])
plt.show()

err3 = fom-rom3
err2 = fom-rom2
err1 = fom-rom1
#relativen Fehler bestimmen
print(f"Relative Hinf error Ordnung 3:   {err3.hinf_norm() / fom.hinf_norm():.3e}")
print(f"Relative Hinf error Ordnung 2:   {err2.hinf_norm() / fom.hinf_norm():.3e}")
print(f"Relative Hinf error Ordnung 1:   {err1.hinf_norm() / fom.hinf_norm():.3e}")
print(f'Relative H2 error Ordnung 3:     {err3.h2_norm() / fom.h2_norm():.3e}')
print(f'Relative H2 error Ordnung 2:     {err2.h2_norm() / fom.h2_norm():.3e}')
print(f'Relative H2 error Ordnung 1:     {err1.h2_norm() / fom.h2_norm():.3e}')
print(f'Relative Hankel error Ordnung 3: {err3.hankel_norm() / fom.hankel_norm():.3e}')
print(f'Relative Hankel error Ordnung 2: {err2.hankel_norm() / fom.hankel_norm():.3e}')
print(f'Relative Hankel error Ordnung 1: {err1.hankel_norm() / fom.hankel_norm():.3e}')

Hinf = [err1.hinf_norm() / fom.hinf_norm(), err2.hinf_norm() / fom.hinf_norm(), err3.hinf_norm() / fom.hinf_norm()]
H2 = [err1.h2_norm() / fom.h2_norm(), err2.h2_norm() / fom.h2_norm(), err3.h2_norm() / fom.h2_norm()]
Hankel = [err1.hankel_norm() / fom.hankel_norm(), err2.hankel_norm() / fom.hankel_norm(), err3.hankel_norm() / fom.hankel_norm()]

fig, ax = plt.subplots()
ax.plot(range(1, len(Hinf)+1), Hinf)
ax.plot(range(1, len(H2)+1), H2)
ax.plot(range(1, len(Hankel)+1), Hankel)
ax.legend(['H infinity', 'H2 Error', 'Hankel Error'])
ax.set_xlabel('Order of reduced system, r')
ax.set_ylabel('Relativ Error')
plt.show()