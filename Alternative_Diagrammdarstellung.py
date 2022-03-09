import control
'''System erster Ordnung: Ex(abgeleitet)=Ax+bu; y=Cx+Du
u=Input, x=state/Zustandsvariable, y=Output
nicht parametrisches Modell, jedoch auch parametrische LTI Systeme sind denkbar über das Argument `mu´

Beispiel ist eine eindimensionale Wärmegleichung mit zwei Inputs und drei Outputs (S. https://docs.pymor.org/2021-2-0/tutorial_lti_systems.html)
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel

plt.rcParams['axes.grid'] = True

#Systemmatrizen bilden
k = 50
n = 2 * k + 1

E = sps.eye(n, format='lil')
E[0, 0] = E[-1, -1] = 0.5
E = E.tocsc()

d0 = n * [-2 * (n - 1)**2]
d1 = (n - 1) * [(n - 1)**2]
A = sps.diags([d1, d0, d1], [-1, 0, 1], format='lil')
A[0, 0] = A[-1, -1] = -n * (n - 1)
A = A.tocsc()

B = np.zeros((n, 2))
B[:, 0] = 1
B[0, 0] = B[-1, 0] = 0.5
B[0, 1] = n - 1

C = np.zeros((3, n))
C[0, 0] = C[1, k] = C[2, -1] = 1

# LTI Modell aus den Matrizen erstellen
fom = LTIModel.from_matrices(A, B, C, E=E)
'''Modellordnung, Inputs, Outputs und weitere Infos abrufen Über
print(fom)
Einzelne Matrizen anzeigen
print(fom.D)
'''

'''Transferfunktion H / Übertragungsfunktion erstellen
H des LTI Systems über die folgenden Befehle berechenbar:
print(fom.transfer_function.eval_tf(0))
print(fom.transfer_function.eval_tf(1))
print(fom.transfer_function.eval_tf(1j))'''

H=fom.transfer_function.eval_tf

#print('\n', H, '\n')
'''Amplituden und Bode plots erstellen
'''
# Amplituden Diagramm
w = np.logspace(-2, 8, 300) # Omega/ Frequenz auf der x Achse
#_ = fom.transfer_function.mag_plot(w, Hz = True)
fig, ax = plt.subplots()
fom.transfer_function.mag_plot(w, Hz= True, ax=ax)
#plt.show()      #Erzeugt das Diagramm

# Bode Diagramm
#_ = fom.transfer_function.bode_plot(w)
fig, ax = plt.subplots()
fom.transfer_function.bode_plot(w)
#plt.show()      #Erzeugt Bode Diagramme

'''Gramsche Matrizen erzeugen über'''
Gramsche_Matrizen = fom.gramian('c_lrcf')
#print(Gramsche_Matrizen)
#print(Gramsche_Matrizen.__len__())
#G = Gramsche_Matrizen.to_numpy()
#print(G.shape)
#print(G)
'''Hankel Eigenwerte
Plot der Hankel EIgenwerte gibt die Güte der Approximation an'''
hsv = fom.hsv()
fig, ax = plt.subplots()
ax.semilogy(range(1, len(hsv)+1), hsv, '.-')
_ = ax.set_title('Hankel singular values')
plt.show()
xWerte=list(range(1, len(hsv)+1))
yWerte= (hsv)
plt.semilogy()
plt.plot(xWerte, yWerte)    #Erzeugt Punkte
plt.scatter(xWerte, yWerte)
plt.xlabel(r'$Hankel singular values$')
plt.show()
#fig.savefig('yourfilename.png')        # Erzeugt ein Bild des Grapfehns
#plt.show()        #Erzeugt die Hankel Eigenwerte

'''Der Abschnitt zu System norms / H2 Glieder wird hier nicht weiter ausgeführt '''
#print(fom.h2_norm())
#print(fom.hinf_norm())
#print(fom.hankel_norm())