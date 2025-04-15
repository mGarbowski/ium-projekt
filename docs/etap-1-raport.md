# Etap 1 - raport

* Definicja problemu biznesowego, zdefiniowanie zadania/zadań modelowania i wszystkich założeń, zaproponowania kryteriów sukcesu)
* Analiza danych z perspektywy realizacji tych zadań (trzeba ocenić, czy dostarczone dane są wystarczające – może czegoś brakuje, może coś trzeba poprawić, domagać się innych danych, ...)

## Temat projektu

"Niewątpliwie nasza wyszukiwarka ofert jest najlepsza w branży, jednakże jakiś czas temu okazało się, że nowo dodane oferty są słabo pozycjonowane przez silnik wyszukujący. Czy da się coś z tym zrobić?"


## Definicja problemu biznesowego

### Problem biznesowy

* Opracowanie sposobu pozycjonowania w wynikach wyszukiwania ofert, które nie mają jeszcze żadnych opinii
  * obecnie oferty bez opinii nie są w ogóle pozycjonowane
  * nie ma punktu odniesienia dla naszego rozwiązania

### Założenia

* Wyszukiwarka działa dobrze dla ofert, które mają jakieś oceny
* W zbiorze danych `listings.csv` mamy parametry ofert oraz ich średnie oceny w różnych kategoriach
* Do pozycjonowania ofert służy średnia ocena ze wszystkich kategorii - do weryfikacji!
* Dla użytkownika średnia ocena jest bardzo dobrym kryterium jakości oferty
* Możemy przewidzieć oceny dla tych ofert, gdzie ich brakuje
* Chcemy, żeby system mając dwie oferty - jedna z rzeczywistą średnią ocen $y$, druga z przewidzianą średnią oceną $y$ wyżej pozycjonował ofertę z rzeczywistą średnią oceną $y$

### Zadanie modelowania

* Zadanie regresji - przewidywanie średniej oceny oferty na podstawie parametrów oferty
* Cel - średnia ocena - średnia z kolumn:
  * `review_scores_rating`
  * `review_scores_accuracy`
  * `review_scores_cleanliness`
  * `review_scores_checkin`
  * `review_scores_communication`
  * `review_scores_location`
  * `review_scores_value`
* Predykcję oceny możemy przemnożyć przez wagę $\in (0, 1]$ żeby oferty o rzeczywistej średniej ocenie jednakowej wartości były lepiej pozycjonowane

### Biznesowe kryteria sukcesu

* Wzrost liczby wyświetleń ofert bez dodanych opinii do poziomu $\ge 50\%$ średniej liczby wyświetleń ofert z dodanymi opiniami

## Analiza danych

### Dostępne zbiory danych

* `calendar.csv`
  * kalendarz dostępności ofert
  * nieprzydatny do naszego zadania
* `reviews.csv`
  * tekstowe opinie do ofert
  * nieprzydatne do naszego zadania
* `listings.csv`
  * parametry ofert
  * wiele atrybutów, które mogą posłużyć za wejście modelu
  * średnie oceny ofert w różnych kategoriach
  * możemy potraktować średnią z ocen we wszystkich kategoriach jako cel regresji

### Analiza zbioru `listings.csv`