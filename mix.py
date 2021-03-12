import neurolab as nl
import numpy as np
import pylab as pl

#Tworzymy zbiór ucząc:
#Zwraca liczbę równomiernie rozmieszczonych próbek,
#obliczanych w przedziale [start, stop, ilosc probek].
x = np.linspace(-7, 7, 30) # os x
# y = np.cos(x) * np.sin(x) # os y
y = np.cos(x) * np.power(x, 3) # os y
y /= np.linalg.norm(y)
size = len(x)
# 20 nowych elementow po jednym elemencie
inp = x.reshape(size,1)
print (inp)
tar = y.reshape(size,1)
print (tar)
# Tworzymy sieć z dwoma warstwami, inicjalizowaną w #sposób losowy, pierwszej warstwie 5 neuronów, w drugiej 1
net = nl.net.newff([[-7,7]],[13,7,1])
# #Uczymy sieć, wykorzystujemy metodę największego spadku gradientu
#(algorytm uczenia)Gradient descent backpropogation
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=500, show=100, goal=0.01)
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
x2 = np.linspace(-7.0,7.0,30)
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

