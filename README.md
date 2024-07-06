# Doggy monitor

## Założenia projektowe
Celem jest stworzenie prototypu urządzenia umożliwiającego monitorowanie psa i jego emocji w trakcie nieobecności właściciela. Powinno ono spełniać następujące kryteria:
1. Rozpoznawanie emocji na podstawie postawy psa;
    - rasy psów zostaną podzielone na grupy według charakterystycznych cech wyglądu (np. kształt uszu, długość kufy)
    - dla minimum trzech grup zostaną wytrenowane osobne modele (najprawdopodobniej ViT) do wykrywania punktów charakterystycznych oraz algorytmy, które na ich podstawie będą obliczać emocje (skuteczność wykrywania punktów i przewidywania emocji: minimum 70% oraz pozytywna opinia behawiorysty/innego specjalisty)
2. Narzędzie śledzące psa;
    - telefon (lub kamera) na obrotowym stojaku (2 dof, zakres obrotów: 90° - stojak ustawiony na podwyższeniu (np. stół) przy ścianie pomieszczenia będzie umożliwiał obrót telefonu do boków, góry i dołu o 45°)
    - Raspberry Pi jako urządzenie sterujące serwami i przesyłające obraz z telefonu na serwer (DigitalOcean)
3. Śledzenie pozycji psa;
    - wykorzystanie filtru kalmana do estymowania przyszłych pozycji psa
    - skuteczność śledzenia psa: minimum 65%
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
