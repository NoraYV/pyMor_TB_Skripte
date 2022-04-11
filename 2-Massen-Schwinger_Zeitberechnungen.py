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

#Berechnungszeit durch zehn Berechnungen der durchschnitt bilden
Summe1 = 0
for i in range(0,100):
    start_fom = time.perf_counter()
    fom = LTIModel.from_matrices(A, B, C, E=E)
    end_fom = time.perf_counter()
    Berechnungszeit_fom_erstellen = end_fom-start_fom
    Summe1 = Berechnungszeit_fom_erstellen + Summe1
print('Durchschnittszeit zum Erstellen des fom', Summe1/100)

fom = LTIModel.from_matrices(A, B, C, E=E)
#print(fom)
#Reductor schaffen
bt = BTReductor(fom)
#print('bt: ', '\n', bt)

Cholskey_factor = bt._gramians()    #low-ranked gramians
#print('Cholskey Faktoren:', Cholskey_factor)


#Berechnungszeit durch zehn Berechnungen der durchschnitt bilden
Summe2 = 0
for i in range(0,100):
    start_rom = time.perf_counter()
    rom = bt.reduce(2)
    end_rom = time.perf_counter()
    Berechnungszeit_rom_erstellen = end_rom-start_rom
    Summe2 = Summe2 + Berechnungszeit_rom_erstellen
print('Durchschnittszeit zum Erstellen des rom', Summe2/100)

rom = bt.reduce(2)
#print('FOM:', fom,'\n', 'ROM',  rom)

w = np.logspace(-2, 8, 300)

# Berechnung des Bode Diagramms für fom, zehn Durchläufe und daraus den Durchschnitt bilden
fom_transferfunction = fom.transfer_function
#print(fom_transferfunction.__init__)
#https://github.com/pymor/pymor/blob/2021.2.0/src/pymor/models/transfer_function.py
#print(fom_transferfunction.eval_tf(1))
Summe3 = 0
for i in range(0,100):
    start_fom = time.perf_counter()
    fom_transferfunction = fom.transfer_function
    end_fom = time.perf_counter()
    Berechnungszeit_fom_erstellen = end_fom - start_fom
    Summe3 = Summe3 + Berechnungszeit_fom_erstellen
print('Durchschnittszeit zum Erstellen der fom Transferfunktion', Summe3/100)

# Berechnung des Bode Diagramms für rom, zehn Durchläufe und daraus den Durchschnitt bilden
Summe4 = 0
for i in range(0,100):
    start_rom = time.perf_counter()
    rom_transferfunction = rom.transfer_function
    end_rom = time.perf_counter()
    Berechnungszeit_rom_erstellen = end_rom - start_rom
    Summe4 = Berechnungszeit_rom_erstellen + Summe4
print('Durchschnittszeit zum Erstellen der rom Transferfunktion', Summe4/100)
