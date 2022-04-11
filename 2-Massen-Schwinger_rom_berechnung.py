import statistics
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor, LQGBTReductor, BRBTReductor
import time
import sys

plt.rcParams['axes.grid'] = True
k1 = 10
k2 = 10
d1 = 1
d2 = 1

E= np.array([[1,0,0,0], [0,1,0,0], [0, 0, 1, 0], [0, 0, 0, 1]])
A= np.array([[0,0,1,0], [0,0,0,1], [-k1-k2, k2, -d1-d2, -d2], [k2, -k2, -d2, d2]])
B = np.array([[0], [0], [0], [1]])
C = np.array([[1,0,0,0]])

fom = LTIModel.from_matrices(A, B, C, E=E)
print('FOM', fom)

#Reductor schaffen; Hier gibt es Problemen: lycotResultWarning: DICO = 'C' and the pencil A - lambda * E has a degenerate pair of eigenvalues. That is, lambda_i = -lambda_j for some i and j, where lambda_i and lambda_j are eigenvalues of A - lambda * E. Hence, equation (1) is singular;  perturbed values were used to solve the equation (but the matrices A and E are unchanged).
bt = BTReductor(fom)
#print('bt: ', '\n', bt)

Cholskey_factor = bt._gramians()    #low-ranked gramians
#print('Cholskey Faktoren BTR:', Cholskey_factor)

''' Andere Reductoren/ Berechnungsarten der Gramsche Matrizen bzw. Cholskey Matrizen
bt = LQGBTReductor(fom)
Cholskey_factor = bt._gramians()
print('Cholskey Faktoren LQGR:', Cholskey_factor)

bt = BRBTReductor(fom)
Cholskey_factor = bt._gramians()
print('Cholskey Faktoren BRR:', Cholskey_factor)
'''

#Pole des Systems
poles = fom.poles()
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
_ = ax.set_title('Poles')
#plt.show()

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
#plt.show()

#Reduktion auf ein System 2. Ordnung
rom = bt.reduce(2)
#print('FOM:', fom,'\n', 'ROM',  rom)

w = np.logspace(-2, 8, 300)

fig, axs = plt.subplots(2, 1, figsize=(6, 12), sharex=True, constrained_layout=True, squeeze=False)
#fig, ax = plt.subplots()
fom.transfer_function.bode_plot(w, ax=axs)
#plt.show()
#fig, ax = plt.subplots()
_ = rom.transfer_function.bode_plot(w, linestyle='--', ax=axs)
plt.legend(['fom', 'rom'])
#plt.show()


#Fehler beider Systeme im Bode Diagramm abbilden
#err = fom-rom
#_ = err.transfer_function.bode_plot(w)
#plt.show()
err = fom-rom
#print('Error System', err)
#print(err.__init__)
#print(err.__str__)
#relativen Fehler bestimmen
print(f"Relative Hinf error:   {err.hinf_norm() / fom.hinf_norm():.3e}")
print(f'Relative H2 error:     {err.h2_norm() / fom.h2_norm():.3e}')
print(f'Relative Hankel error: {err.hankel_norm() / fom.hankel_norm():.3e}')
