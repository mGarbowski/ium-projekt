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

#### Wielkość zbioru danych
* Wiersze: 4195
* Wiersze z ocenami: 3223

#### Wszystkie atrybuty

* `id`
* `listing_url`
* `scrape_id`
* `last_scraped`
* `source`
* `name`
* `description`
* `neighborhood_overview`
* `picture_url`
* `host_id`
* `host_url`
* `host_name`
* `host_since`
* `host_location`
* `host_about`
* `host_response_time`
* `host_response_rate`
* `host_acceptance_rate`
* `host_is_superhost`
* `host_thumbnail_url`
* `host_picture_url`
* `host_neighbourhood`
* `host_listings_count`
* `host_total_listings_count`
* `host_verifications`
* `host_has_profile_pic`
* `host_identity_verified`
* `neighbourhood`
* `neighbourhood_cleansed`
* `neighbourhood_group_cleansed`
* `latitude`
* `longitude`
* `property_type`
* `room_type`
* `accommodates`
* `bathrooms`
* `bathrooms_text`
* `bedrooms`
* `beds`
* `amenities`
* `price`
* `minimum_nights`
* `maximum_nights`
* `minimum_minimum_nights`
* `maximum_minimum_nights`
* `minimum_maximum_nights`
* `maximum_maximum_nights`
* `minimum_nights_avg_ntm`
* `maximum_nights_avg_ntm`
* `calendar_updated`
* `has_availability`
* `availability_30`
* `availability_60`
* `availability_90`
* `availability_365`
* `calendar_last_scraped`
* `number_of_reviews`
* `number_of_reviews_ltm`
* `number_of_reviews_l30d`
* `first_review`
* `last_review`
* `review_scores_rating`
* `review_scores_accuracy`
* `review_scores_cleanliness`
* `review_scores_checkin`
* `review_scores_communication`
* `review_scores_location`
* `review_scores_value`
* `license`
* `instant_bookable`
* `calculated_host_listings_count`
* `calculated_host_listings_count_entire_homes`
* `calculated_host_listings_count_private_rooms`
* `calculated_host_listings_count_shared_rooms`
* `reviews_per_month`

#### Całkiem nieprzydatne atrybuty

Atrybuty, które możemy odrzucić, bo nie niosą informacji, są bardzo niezbalansowanymi kategoriami,
mają nieprzydatny format itd.

* `id`
  * nie niesie informacji o jakości
* `listing_url`
  * powiązany z id
* `scrape_id`
  * nie powinno być istotne, kiedy zostały zebrane dane
* `last_scraped`
  * wynika ze `scrape_id`
* `source`
  * związane z procesem zbierania danych
* `picture_url`
  * nie mamy jak tego wykorzystać, chyba że pobierać każdy i analizować metodami CV
  * brak obrazka może być istotny
  * w zbiorze nie ma przykładów z brakującym obrazkiem
* `host_id`
  * nie niesie informacji o jakości
* `host_url`
    * nie niesie informacji o jakości
* `host_name`
    * nie niesie informacji o jakości
* `host_location`
  * miasto i państwo lub miasto i stan USA
  * 200 różnych
  * ~75% to `Berlin, Germany`
  * można wyciągnąć państwo
  * 42 różne
  * ponad 3000 to Niemcy, kolejne USA ma 28
  * raczej nieprzydatne, bo dużo unikalnych wartości i jedna wyraźnie dominuje
* `host_thumbnail_url`
  * brak miniaturki może być istotny
  * tylko 3 braki w całym zbiorze
* `host_picture_url`
  * brak zdjęcia może być istotny - jest na to oddzielny atrybut
  * tylko 3 braki w całym zbiorze
* `host_neighbourhood`
  * 104 unikalne wartości - raczej za dużo
* `neighbourhood`
  * nominalny
  * bezużyteczny - `Berlin, Germany` występuje 2043 razy
  * każda z pozostałych 12 kategorii też zawiera `Berlin, Germany` i występuje po 1 raz
* `neighbourhood_cleansed`
  * 133 unikalne wartości - raczej za dużo
  * `neighbourhood_group_cleansed` jest powiązany i powinien być bardziej informatywny
* `latitude`
  * surowe współrzędne nie będą użyteczne przy regresji
  * `neighbourhood_group_cleansed` zawiera informację o lokalizacji
* `longitude`
  * surowe współrzędne nie będą użyteczne przy regresji
  * `neighbourhood_group_cleansed` zawiera informację o lokalizacji
* `amenities`
  * lista wszystkich udogodnień
  * potencjalnie bardzo przydatne
  * unikalnych wartości w listach jest ponad 1000
  * można spróbować one-hot encoding i zamienić na zanurzenie w modelu neuronowym
* `calendar_updated`
  * braki we wszystkich wierszach
* `calendar_last_scraped`
  * związane z procesem zbierania danych
* `first_review`
  * nie będzie dostępny dla nowych ofert
* `last_review`
  * nie będzie dostępny dla nowych ofert
* `reviews_per_month`
  * nie będzie dostępny dla nowych ofert


#### Atrybuty tekstowe

Atrybuty nieprzydatne w takiej postaci, ale można by je przetworzyć przez LLM i uzyskać zanurzenia

* `name`
  * można by wyciągnąć informację o typie oferty, ale na to już jest atrybut
* `description`
  * potencjalnie bardzo przydatne
  * można by utworzyć zanurzenie jakimś zewnętrznym modelem
  * 185 braków
* `neighborhood_overview`
  * ~50% braków
* `host_about`
  * ~50% braków
  * wątpliwe czy w ogóle zawiera przydatne informacje
* `license`
  * ~40% braków
  * wątpliwe czy w ogóle zawiera przydatne informacje