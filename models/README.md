## Trenowanie modelu

1. Biblioteki zostały zapisane w [requirements.txt](./requirements.txt). Ich wersje są poprawne pod warunkiem zainstalowania cuda toolkit w wersji 12.4.

2. Przed rozpoczęciem trenowania należy upewnić się, że projekt został przygotowany w następujący sposób:
- w projekcie znajdują się pliki [train.py](./train.py) i [config.yaml](./config.yaml) oraz folder o nazwie "datasets".
- w "datasets" znajduje się folder "data", a w nim "images" i "labels", wewnątrz których znajdują się dane podzielone na części treningową, walidacyjną i testową.

3. Po zakończeniu trenowania należy przetestować model programem [test_model.py](./test_model.py). Model może zostać zaakceptowany pod warunkiem uzyskania minimum 75% skuteczności (accuracy, nie precision).

4. Pliki [rysiowanie.py](./rysiowanie.py) i [test_for_one_image.py](./test_for_one_image.py) są pomocniczymi do testów.
