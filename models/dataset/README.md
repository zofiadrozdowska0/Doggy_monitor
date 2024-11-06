## O datasecie 

Podział ras psów - zgromadzono zdjęcia reprezentujące wszystkie rasy psów istniejące w rejestrze FCI. Dopasowano rasy do siebie biorąc pod uwagę kształt uszu, pyska i ogona, gęstość i długość sierści oraz długość łap w stosunku do długości tułowia.  Wybrano 3 grupy: 
- Klapnięte uszy, prosty ogon, średniej długości sierść, długi pysk, długie nogi (1)
- Postawione uszy, prosty ogon, gęsta sierść średniej długości, długi pysk, długie nogi (2)
- Klapnięte uszy, prosty ogon, opływowy kształt, krótka sierść, długi pysk, długie nogi (3)

W celu równomiernego rozłożenia liczby filmów dla danej rasy w grupie, a jednocześnie biorąc pod uwagę rzadkość występowania niektórych z nich, opisane wcześniej grupy podzielono na mniejsze na podstawie kolorów sierści, faktury włosa i innych bardziej szczególnych cech. Przypisanie ras do podzbiorów opisano w pliku [Wybrane_grupy_psów.pdf](./Wybrane_grupy_psów.pdf). Zebrano po 60 filmów dla grup 1A-E i 3A-E oraz po 150 dla grup 2A i 2B.

Do anotacji wykorzystano narzędzie CVAT. Anotowane punkty zapisano w pliku nasz_szkieleton.txt. 
Filmy oznaczano na co dziesiątej klatce, a dataset eksportowano w formacie yolov8-pose zachowując opisane klatki.

W zbieraniu filmów udział brali: Zofia Drozdowska, Joanna Ryś, Tomasz Sekrecki

W anotacji udział brali: Zofia Drozdowska, Joanna Ryś, Tomasz Sekrecki, Michał Kruszewski

Uporządkowane i połączone datasety dla każdej grupy: (todo)
- 1
- 2
- 3

Programy pomocnicze do tworzenia datasetu:
- [data_merge.py](./data_merge.py) - łączenie folderów z danymi
- [norm.py](./norm.py) - poprawa zapisu danych po pobraniu z CVATa
- [split_data.py](./split_data.py) - podział na dane treningowe (70%), walidacyjne (20%) i testowe (10%)
- [revert_split.py](./revert_split.py) - przywrócenie datasetu do stanu sprzed podziału
- [zip_unpacker.py](./zip_unpacker.py) - hurtowe rozpakowywanie zipów
