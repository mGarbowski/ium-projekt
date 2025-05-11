# Inżynieria Uczenia Maszynowego - projekt

* Mikołaj Garbowski
* Maksym Bieńkowski

## Temat

### Problem

"Niewątpliwie nasza wyszukiwarka ofert jest najlepsza w branży, jednakże jakiś czas temu okazało się, że nowo dodane oferty są słabo pozycjonowane przez silnik wyszukujący. Czy da się coś z tym zrobić?"

### Koncepcja rozwiązania

* Zadanie regresji
* Skoro wyszukiwarka działa dobrze dla ofert, które mają jakieś oceny, to spróbujmy przewidzieć oceny tam gdzie ich brakuje
* Model regresji, gdzie celem jest średnia ocena

### Kontekst

W ramach projektu wcielamy się w rolę analityka pracującego dla portalu Nocarz - serwisu, w
którym klienci mogą wyszukać i dokonać rezerwacji noclegu (zapewnianego przez oferentów).
Praca na tym stanowisku nie jest łatwa - zadanie dostajemy w formie enigmatycznego opisu i to
do nas należy doprecyzowanie szczegółów tak, aby dało się je zrealizować. To oczywiście
wymaga zrozumienia problemu, przeanalizowania danych, czasami negocjacji z szefostwem.
Poza tym, oprócz przeanalizowania zagadnienia i wytrenowania modeli, musimy przygotować je
do wdrożenia produkcyjnego - zakładając, że w przyszłości będą pojawiać się kolejne ich wersje,
z którymi będziemy eksperymentować.

Jak każda szanująca się firma internetowa, Nocarz zbiera dane dotyczące swojej działalności - są
to (analitycy mogą wnioskować o dostęp do tych informacji na potrzeby realizacji zadania):

* szczegółowe dane o dostępnych lokalach,
* recenzje lokali,
* kalendarz z dostępnością i cenami,
* baza klientów i sesji

## Harmonogram

* Etap 1 2025.04.25
* Etap 2 2025.05.23
* [Opis](./docs/Projekt_IUM25L.pdf)


## Uruchomienie i instalacja

Instalacja bibliotek i wirtualnego środowiska:

```bash
pdm install --dev
```

Uruchomienie aplikacji w trybie deweloperskim:

```bash
pdm run dev
```

Uruchomienie aplikacji w kontenerze Docker:

```bash
docker compose up --build -d
```

Zapisanie logów do pliku:

```bash
docker compose logs > data/service.log
```

Uruchomienie testów:

```bash
pdm run test
```
