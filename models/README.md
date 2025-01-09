## Trenowanie modelu

1. Biblioteki zostały zapisane w [requirements.txt](./requirements.txt). Ich wersje są poprawne pod warunkiem zainstalowania cuda toolkit w wersji 12.4.

2. Przed rozpoczęciem trenowania należy upewnić się, że projekt został przygotowany w następujący sposób:
- w projekcie znajdują się pliki [train.py](./train.py) i [config.yaml](./config.yaml) oraz folder o nazwie "datasets".
- w "datasets" znajduje się folder "data", a w nim "images" i "labels", wewnątrz których znajdują się dane podzielone na części treningową, walidacyjną i testową.

3. Po zakończeniu trenowania należy przetestować model programem [test_model.py](./test_model.py). Model może zostać zaakceptowany pod warunkiem uzyskania minimum 75% skuteczności (accuracy, nie precision).

4. Pliki [rysiowanie.py](./rysiowanie.py) i [test_for_one_image.py](./test_for_one_image.py) są pomocniczymi do testów.

## Rezultaty

Pojęcia: 
- Precision – precyzja; określa, ile z wykrytych obiektów należy do pozytywnych przykładów 
- Recall – czułość; określa, jaki odsetek rzeczywistych obiektów został poprawnie wykryty 
- mAP50 – Mean Average Precision; średnia precyzja przy progu IoU (Intersection over Union) 0.5 dla wykrycia obiektów 
- box loss – strata dotycząca dopasowania granic b-boxa 
- pose loss – strata dotycząca dopasowania parametrów pozycji 
- kobj loss – strata dotycząca wykrywania kluczowych obiektów 
- cls loss – strata dotycząca przypisania wykrytego obiektu do właściwej klasy 
- dfl loss – distance focal loss; strata dotycząca precyzji dopasowania współrzędnych obiektów 
- P lub B przy wykresach precyzji, czułości i mAP50 oznacza, że wykres dotyczy odpowiednio pozycji lub b-boxa. 
- Accuracy – skuteczność – nie jest ona bezpośrednio obliczana w trakcie treningu i walidacji w przypadku modelu YOLO, dlatego napisano skrypt testujący, który porównuje obiekty wykryte przez model z anotacjami. Wykorzystywany do tego jest próg określany jako 7% długości psa w pikselach. Na tej podstawie zliczane są przypadki prawdziwie pozytywne (TP) i negatywne (TN) oraz fałszywie pozytywne (FP) i negatywne (FN), a skuteczność obliczana jest ze wzoru: Accuracy = (TP + TN)/(TP + FP + TN + FN)

### Grupa 1
Accuracy = 76,17%
[wykresy]

### Grupa 2
Accuracy = 75,99%
[wykresy]

### Grupa 3
Accuracy = 76,78%
[wykresy]