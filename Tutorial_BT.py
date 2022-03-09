import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor

plt.rcParams['axes.grid'] = True

k = 50
n = 2*k+1
E = sps.eye(n, format='lil') # Einheitsmatrix
E[0,0] = E[-1, -1] = 0.5 # Erster und letzter Wert werden zu 0,5 gesetzt
E = E.tocsc()

d0 = n*[-2*(n-1)**2]
d1 = (n - 1) * [(n-1)**2]
A = sps.diags([d1, d0, d1], [-1, 0, 1], format='lil')
A[0,0] = A[-1, -1] = -n*(n-1)
A = A.tocsc()

B = np.zeros((n, 2))
B[:, 0] = 1
B[0,0] = B[-1, 0] = 0.5
B[0,1] = n-1

C = np.zeros((3, n))
C[0,0] = C[1, k] = C[2, -1] = 1

fom = LTIModel.from_matrices(A, B, C, E=E)
print(fom)
'''Annahme zur Balanced Truncation:
System erster Ordnung: Ex(abgeleitet)=Ax+bu; y=Cx+Du
andernfalls kann x auch durch Tx(schlange/Nährung) oder andere Multiplikationen genutzt werden.
Weitere Infos: https://docs.pymor.org/2021-2-0/tutorial_bt.html
Wichtig ist dass eine invertierbate Transformationsmatrix T und S der Ordnung nxn exestieren, sodass Nährungen wie folgt entstehen:
E(schlange)=S(transformiert)ET=I(Einheitsmatrix)
A(Schlange)=S(transformiert)AT
B(Schlange)=S(transformiert)B
C(Schlange)=CT
sodass die Gramschen Matrizen P(schlange) und Q(schlange) identisch sind und aus aus den diagonalen Hankel singular Werten entstehen. 
Tutorial für die Gramschen Matrizen: https://docs.pymor.org/2021-2-0/tutorial_lti_systems.html
bzw. als Tutorial_Gramsche_Matrizen abgespeichert'''

#System BR reduzieren
bt = BTReductor(fom)
print(bt, '\n')

error_bounds = bt.error_bounds()
hsv = fom.hsv()
fig, ax = plt.subplots()
ax.semilogy(range(1, len(error_bounds)+1), error_bounds, '.-')
ax.semilogy(range(1, len(hsv)), hsv[1:], '.-')
ax.set_xlabel('Reduced order')
#_ = ax.set_title(r'Upper an lower $\mathcal{H}_\infty$ error bounds')
plt.show()

'''Modell auf eine Gewisse Ordnung reduzieren
Beispiel der Ordnung 10'''
rom = bt.reduce(10)

#print(bt.V, bt.W)      #Erzeugt die Projektionsmatrizen V und W

'''Vergleich der Amplituden und Bode Diagramme des vollständigen und des reduzierten Modells'''
#Amplituden vergleichen
w = np.logspace(-2, 8, 300)
fig, ax = plt.subplots()
fom.transfer_function.mag_plot(w, ax=ax, label='FOM')
rom.transfer_function.mag_plot(w, ax=ax, linestyle='--', label='ROM')
_ = ax.legend()
plt.show()

# Bode Diagramme vergleichen
fig, axs = plt.subplots(6, 2, figsize=(12, 24), sharex=True, constrained_layout=True)
fom.transfer_function.bode_plot(w, ax=axs)
_ = rom.transfer_function.bode_plot(w, ax=axs, linestyle='--')
plt.show()

'''Fehler beider Systeme darstellen'''
err = fom - rom
#Amplituendenfehler
fig, ax = plt.subplots()
_ = err.transfer_function.mag_plot(w)
#Bode Fehler
fig, ax = plt.subplots()
_ = err.transfer_function.bode_plot(w)
#plt.show()

'''relative Fehler bestimmen'''
# Wenn Fehler auftreten, alle plt.show() unkommentieren. Dann läuft das Programm durch
print(f"Relative Hinf error:   {err.hinf_norm() / fom.hinf_norm():.3e}")
print(f'Relative H2 error:     {err.h2_norm() / fom.h2_norm():.3e}')
print(f'Relative Hankel error: {err.hankel_norm() / fom.hankel_norm():.3e}')
