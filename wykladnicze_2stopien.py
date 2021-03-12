import neurolab as nl
import numpy as np
import pylab as pl

#Tworzymy zbiór ucząc:
#Zwraca liczbę równomiernie rozmieszczonych próbek,
#obliczanych w przedziale [start, stop, ilosc probek].
x = np.linspace(-1.6, 1.6, 30) # os x
y = np.power(x, 2)  # os y
y /= np.linalg.norm(y)
size = len(x)
# 20 nowych elementow po jednym elemencie
inp = x.reshape(size,1)
print (inp)
tar = y.reshape(size,1)
print (tar)
# Tworzymy sieć z dwoma warstwami, inicjalizowaną w #sposób losowy, pierwszej warstwie 5 neuronów, w drugiej 1
net = nl.net.newff([[-1.6, 1.6]],[10, 1])
# #Uczymy sieć, wykorzystujemy metodę największego spadku gradientu
#(algorytm uczenia)Gradient descent backpropogation
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=1000, show=100, goal=0.01)
#Symulujemy
out = net.sim(inp)
fig = pl.figure()
fig.subplots_adjust(hspace=0.8, wspace=0.4)
#wykres z iloscia epok i bledem uzyskanym
pl.subplot(211)
pl.plot(error)
pl.grid(color='g', linestyle='--', linewidth=0.5)
pl.xlabel('Numer epoki')
pl.ylabel('blad (domyslnie SSE)')
pl.legend(['blad'])
pl.title('Zmiana bledu')
# #Tworzymy wykres z wynikami
x2 = np.linspace(-1.5,1.5,30)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = out.reshape(size)
#wykres z funkcja wzorcowa i wyuczona
pl.subplot(212)
pl.plot(x2, y2, '.',color='black')
pl.title('Aproksymacja funkcji')
pl.grid(color='g', linestyle='--', linewidth=0.5)
pl.xlabel('os x')
pl.ylabel('os y')
pl.plot(x , y, '.',color='red')
pl.legend(['wynik uczenia','wartosc rzeczywista'])
pl.show()

