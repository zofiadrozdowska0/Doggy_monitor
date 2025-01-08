# Doggy monitor

![](docs/assets/dog.mp4)

## Założenia projektowe

Celem jest stworzenie prototypu urządzenia umożliwiającego monitorowanie psa i jego emocji w trakcie nieobecności właściciela. Powinno ono spełniać następujące kryteria:

1. Rozpoznawanie emocji na podstawie postawy psa;
   - rasy psów zostaną podzielone na grupy według charakterystycznych cech wyglądu (np. kształt uszu, długość kufy)
   - dla minimum trzech grup zostaną wytrenowane osobne modele (najprawdopodobniej ViT) do wykrywania punktów charakterystycznych oraz algorytmy, które na ich podstawie będą obliczać emocje (skuteczność wykrywania punktów i przewidywania emocji: minimum 75% oraz pozytywna opinia behawiorysty/innego specjalisty)
2. Narzędzie śledzące psa;
   - telefon (lub kamera) na obrotowym stojaku (2 dof, zakres obrotów: 90° - stojak ustawiony na podwyższeniu (np. stół) przy ścianie pomieszczenia będzie umożliwiał obrót kamery do boków, góry i dołu o 45°)
   - Raspberry Pi jako urządzenie sterujące serwami i przesyłające obraz z kamery na serwer (DigitalOcean)
3. Śledzenie pozycji psa;
   - wykorzystanie filtru kalmana do estymowania przyszłych pozycji psa
   - skuteczność śledzenia psa: minimum 70%
   - maksymalna odległość psa od urządzenia: 3m
4. Zastosowanie serwera;
   - odbiór obrazu z RPI i przetworzenie go za pomocą modeli i algorytmów
   - przesłanie obrazu na telefon właściciela psa wraz z przewidywanymi emocjami
5. Aplikacja dla właściciela psa;
   - umożliwia odczyt emocji docelowo w czasie rzeczywistym, jednak należy uwzględnić opóźnienia zależne od czasu trwania przetwarzania danych
   - zapis historii emocji wraz z obrazem o zmniejszonej jakości

<B>Stack technologiczny:</B>

- Python; pytorch (Modified BSD License), OpenCV (Apache 2 License)
- Kotlin
- RPI
- telefon (lub kamera)

<B>Wagi założeń projektowych:</B>

- Rozpoznawanie emocji - 25%
- Narzędzie śledzące - 20%
- Śledzenie pozycji - 25%
- Aplikacja - 15%
- Integracja wszystkich elementów - 10%

## Harmonogram

| Nazwa zadania                                 | Czas pracy    | Liczba osób |
| --------------------------------------------- | ------------- | ----------- |
| Zbieranie i wstępne przetwarzanie danych      | Tydzień 1 i 2 | 4           |
| Prototypowanie mechanizmu śledzenia           | Tydzień 1 i 2 | 3           |
| Sprawdzenie możliwości back-endowych          | Tydzień 1 i 2 | 3           |
| Rozwój modeli rozpoznawania emocji            | Tydzień 3 i 4 | 5           |
| Implementacja filtru Kalmana                  | Tydzień 3 i 4 | 3           |
| Prototyp front-endu dla aplikacji właściciela | Tydzień 3 i 4 | 2           |
| Testowanie i walidacja modeli (emocji)        | Tydzień 5 i 6 | 4           |
| Testowanie i walidacja algorytmu śledzenia    | Tydzień 5 i 6 | 4           |
| Komunikacja aplikacji i integracja            | Tydzień 5 i 6 | 2           |
| Ostateczna optymalizacja narzędzia            | Tydzień 7 i 8 | 3           |
| Pełna kalibracja urządzenia i testy końcowe   | Tydzień 7 i 8 | 3           |
| Dodatkowe poprawki                            | 2 tygodnie    | -           |

<B>Zbieranie i wstępne przetwarzanie danych</B>

- Zebranie 5-sekundowych filmów dla minimum 3 grup ras psów, zwracając uwagę na cechy charakterystyczne wyglądu.
- Wstępne przetwarzanie danych: anotacja obrazów i czyszczenie danych.
- <B>Kryterium akceptacji:</B> 300 filmów dla każdej z grup, równomiernie dla każdej z ras.

<B>Prototypowanie mechanizmu śledzenia</B>

- Konfiguracja Raspberry Pi, integracja kamery i sterowanie serwomechanizmem do obrotu.
- Sterowanie - najpierw prosty program dowolnie obracający kamerę, potem algorytm podążający za psem.
- <B>Kryterium akceptacji:</B> Mechanizm ma 2 stopnie swobody, kamera może obracać się o 45° w górę, w dół i na boki.

<B>Sprawdzenie możliwości back-endowych</B>

- Sprawdzenie rozwiązań dotyczących serwera i przesyłu danych.
- <B>Kryterium akceptacji:</B> Dokładne opracowanie i przedstawienie planu integracji urządzeń.

<B>Rozwój modeli rozpoznawania emocji</B>

- Trenowanie modeli ViT oraz opracowanie wstępnego algorytmu odczytującego emocje.
- Sprawdzenie możliwości konsultacji z behawiorystą.
- <B>Kryterium akceptacji:</B> 75% skuteczności każdego z modeli ViT.

<B>Implementacja filtru Kalmana</B>

- Wdrożenie filtru Kalmana w celu szacowania przyszłych ruchów psa, aby zapewnić nadążanie kamery za zwierzęciem.
- <B>Kryterium akceptacji:</B> 70% skuteczności w przewidywaniu przyszłych pozycji.

<B>Prototyp front-endu dla aplikacji właściciela</B>

- Zaprojektowanie i stworzenie podstawowej aplikacji dla właściciela psa.
- <B>Kryterium akceptacji:</B> nie dotyczy.

<B>Testowanie i walidacja modeli (emocji)</B>

- Udoskonalenie algorytmów w celu optymalizacji działania.
- <B>Kryterium akceptacji:</B> nie dotyczy.

<B>Testowanie i walidacja algorytmu śledzenia</B>

- Przetestowanie wydajności śledzenia i obracania się kamery, optymalizacja i naprawa ewentualnych błędów.
- <B>Kryterium akceptacji:</B> Kamera płynnie obraca się, pies znajduje się w środkowej części kadru.

<B>Komunikacja aplikacji i integracja</B>

- Integracja back-endu z aplikacją, aby wyświetlać emocje i przesyłać obrazy.
- <B>Kryterium akceptacji:</B> Przesyłanie obrazu w czasie rzeczywistym, dopuszczalne opóźnienie w ocenie emocji maksymalnie kilka minut.

<B>Ostateczna optymalizacja narzędzia, pełna kalibracja urządzenia i testy końcowe</B>

- Ostatnie zadania odnoszą się do poprawy działania narzędzia oraz naprawienia wszystkich błędów napotkanych w trakcie pracy nad projektem.
