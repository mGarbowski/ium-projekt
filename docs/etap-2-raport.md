# Etap 2 – Raport


- dokumentacja procesu budowy modelu oraz porównanie wyników,
- mikroserwis umożliwiający serwowanie predykcji,
- przeprowadzenie testu A/B,
- materiały potwierdzające skuteczność implementacji.

---

## Temat projektu

„Niewątpliwie nasza wyszukiwarka ofert jest najlepsza w branży, jednakże jakiś czas temu zauważyliśmy, że nowo dodane oferty są słabo pozycjonowane przez silnik wyszukiwania. Czy można to poprawić?”

---

## Proces budowy modelu i porównanie wyników

### Model bazowy

Jako model bazowy zastosowano prosty mechanizm predykcji oparty na rozkładzie normalnym, którego parametry (średnia i odchylenie standardowe) zostały wyliczone na podstawie dostarczonego zbioru danych, z uwzględnieniem późniejszego rozszerzenia tego zbioru.

### Model docelowy

Model docelowy to prosta sieć neuronowa wykonująca regresję liniową. Jej hiperparametry zostały dobrane metodą grid search z wykorzystaniem platformy Weights & Biases (skrypt: `src/model/neural_net/sweep.py`).

Główne cechy modelu:
- 62 cechy wejściowe (opisane w poprzednim raporcie),
- 3 warstwy ukryte,
- wczesne zatrzymywanie treningu przy wykryciu stagnacji na zbiorze walidacyjnym,
- techniki preprocesingu: dropout, normalizacja i imputacja danych (bez batch normalizacji),
- stopniowe zmniejszanie learning rate w trakcie treningu.

W trakcie eksperymentów testowano różne liczby i rozmiary warstw, funkcje aktywacji, wartości learning rate, współczynniki dropoutu oraz obecność/nieobecność batch normalizacji. Ostatecznie wybrany model osiągnął stratę na zbiorze walidacyjnym na poziomie `0.06`, co stanowi poprawę względem lasu losowego, który osiągał wynik `0.07`.

---

## Testy A/B i ich wyniki

Testy A/B zrealizowano poprzez przypisywanie użytkowników do wariantów testowych z wykorzystaniem UUID i consistent hashing. Skrypt `src/scripts/mock_app_requests.py` odpowiada za generowanie i przesyłanie zapytań testowych do mikroserwisów.

Dane logowane przez serwer zapisywane są do pliku, który następnie analizowany jest w notebooku `notebooks/analyze_logs.ipynb`. Tam również znajduje się porównanie skuteczności obu modeli.

---

## Implementacja mikroserwisu

Mikroserwis został zrealizowany jako aplikacja FastAPI (`src/app.py`). Obsługuje on zapytania predykcyjne przy użyciu modułu `PredictionService`, który korzysta z instancji klasy `AvgRatingPredictionModel`.

Zaimplementowano kilka wariantów modeli:
- losowy,
- regresji liniowej,
- sieci neuronowej (model docelowy).

---

## Weryfikacja działania implementacji

Za dowody skuteczności implementacji uznajemy:
- logi mikroserwisu,
- testy jednostkowe,
- notebook z porównaniem wyników modeli.

---

## Odstępstwa od Raportu 1

Zrezygnowaliśmy z wykorzystania atrybutu `avg_rating_by_host`, mimo tego, jak obiecujący wydawał się początkowo. Był obliczany w niepoprawny sposób, co prowadziło do wycieku danych – zawierał on wartość `avg_rating` z tego samego wiersza, co zaburzało wiarygodność predykcji, zwłaszcza w przypadku ogłoszeń pochodzących od hostów z tylko jednym listingiem.

Ze względu na to, że aż 80% ogłoszeń należało do takich hostów (`notebooks/exploration_listings.ipynb`), imputacja wartości dla pozostałych przypadków byłaby niewiarygodna i mogłaby zafałszować wyniki modelu.
