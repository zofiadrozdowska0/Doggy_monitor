## Serwer
1. Biblioteki zostały zapisane w [requirements.txt](./requirements.txt). 

### Algorytm

Skrypt wstępnie przetwarza otrzymaną klatkę aby dostosować ją do używanego modelu sieci neuronowej. Następnie klatka zostaje przekazana do sieci i przeprowadzana jest inferencja.

Dane wynikowe z sieci dzielone są na bounding boxy, keypointsy oraz confidence score dla kazdego boxa i punktu. Następnie do algorytmu przekazywane są dane dla obiektu, którego confidence score jest największy. 

Algorytm oblicza kąty poszczególnych części ciała, o ile każdy z punktów należących do danej części ciała jest widoczny.

Na klatce następnie rysowany jest bounding box, a na podstawie wcześniej obliczonych kątów przypisywana jest pozycja poszczególnych części ciała psa.

Nastepnie dla ustalonych pozycji, przy pomocy drzewa decyzyjnego, przypisana zostaje psu konkretna emocja. 

![Drzewo decyzyjne](./drzewo_decyzyjne.png)

Ostateczna emocja wyświetlana użytkownikowi determinowana jest na podstawie ostatnich 10 klatek. Cały skrypt zwraca klatkę z zaznaczonym bounding boxem, dane bounding boxa oraz emocję.
