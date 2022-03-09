from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor

plt.rcParams['axes.grid'] = True

k = 4
E = sps.eye(k, format='lil') # Einheitsmatrix
E = E.tocsc()
A= np.array([[0,0,0,1], [0,0,0,1], [-5, 3, 0, 0], [3, -3, 0, 0]])
B = np.array([[0], [0], [0], [1]])
C = np.array([[0,1,0,0]])

fom = LTIModel.from_matrices(A, B, C, E=E)

#Reductor schaffen
bt = BTReductor(fom)

Cholskey_factor = bt._gramians()    #low-ranked gramians
print('Cholskey Faktoren:', Cholskey_factor)

#Fehlerschranke pro Ordnung
error_bounds = bt.error_bounds()
hsv = fom.hsv()
fig, ax = plt.subplots()
ax.semilogy(range(1, len(error_bounds)+1), error_bounds, '.-')
ax.semilogy(range(1, len(hsv)), hsv[1:], '.-')
ax.legend(['error_bounds', 'hsv'])
ax.set_xlabel('Reduced order')
_ = ax.set_title(r'Upper and lower $\mathcal{H}_\infty$ error bounds')
plt.show()

#Reduktion auf ein System 2. Ordnung
rom = bt.reduce(2)
print(fom, rom)

w = np.logspace(-1, 30, 300)

#fig, axs = plt.subplots(2, 2, figsize=(12, 24), sharex=True, constrained_layout=True)
#fig, ax = plt.subplots()
fom.transfer_function.bode_plot(w)
#fig, ax = plt.subplots()
#_ = rom.transfer_function.bode_plot(w, linestyle='--')
#plt.show()

#Fehler beider Systeme im Bode Diagramm abbilden
#err = fom-rom
#_ = err.transfer_function.bode_plot(w)
#plt.show()'''

#relativen Fehler bestimmen
#print(f"Relative Hinf error:   {err.hinf_norm() / fom.hinf_norm():.3e}")
#print(f'Relative H2 error:     {err.h2_norm() / fom.h2_norm():.3e}')
#print(f'Relative Hankel error: {err.hankel_norm() / fom.hankel_norm():.3e}')
