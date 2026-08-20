# Raw source data

Public source files for the external indicators, archived as retrieved.
Proprietary case data is not held here and is excluded from version control
(see Section 3.5, Reproducibility and ethics).

All files below were retrieved on **20 August 2026**.

## Public-holiday calendar (protocol C1-2)

| File | Source |
|---|---|
| `calendar_2023.json` | https://xmlcalendar.ru/data/ru/2023/calendar.json |
| `calendar_2024.json` | https://xmlcalendar.ru/data/ru/2024/calendar.json |
| `calendar_2025.json` | https://xmlcalendar.ru/data/ru/2025/calendar.json |
| `calendar_2026.json` | https://xmlcalendar.ru/data/ru/2026/calendar.json |

Transferred calendar for the Russian Federation. Day strings are parsed under
protocol row C1-4: a listed day without `*` is non-working; `*` marks a
shortened working day; a weekend absent from the string is a full working day
(reverse transfer). Every non-working day and every reverse transfer was
reconciled against the official production-calendar PDFs (C1-1, C1-3).
General holiday libraries are not used (C1-5).

Parsed by `src/build_holiday_clusters.py`, which writes the 17-cluster set to
`data/holiday_clusters.csv` (C1-6, C1-7). The set is generated, not
transcribed, and is reproduced as Table 3 of the thesis.

## RUB/THB exchange rate (protocol C2-1)

| File | Source |
|---|---|
| `RC_F01_01_2023_T31_07_2026.xlsx` | Bank of Russia, official daily rate, dynamics report |

Requested range 1 January 2023 to 31 July 2026; the series returned begins
19 January 2023, which is earlier than the binding requirement of sample start
minus the longest candidate lag (1 July 2023 − 90 days = 2 April 2023).
874 quoted days. `nominal` is uniformly 10 and `cdx` uniformly `Baht`,
verifying C2-2. The published quote in roubles per 10 baht is rescaled to
roubles per 1 baht (C2-3). Non-trading days take the most recent
determination effective on or before the date (C2-4): 363 of the 1,127
sample days are filled this way, in 141 two-day weekend runs together with
public-holiday runs of up to twelve days across the three New Year periods.

## Yandex Wordstat search interest (protocol C3-1, C3-2)

| File | Query |
|---|---|
| `wordstat_dynamic_туры_в_Таиланд.csv` | туры в Таиланд |
| `wordstat_dynamic_туры_в_Тайланд.csv` | туры в Тайланд |
| `wordstat_dynamic_туры_на_Пхукет.csv` | туры на Пхукет |
| `wordstat_dynamic_туры_в_Паттайю.csv` | туры в Паттайю |

Wordstat query-dynamics report. Metric: number of queries. Frequency: weekly.
Region: Russian Federation. Device type: all devices. Match: Wordstat default
broad match. Each file records these settings, the query and the requested
range in the header row, which is preserved as retrieved.

Downloaded range 26 December 2022 to 2 August 2026; 188 weekly observations
per query on a common Monday index with no gaps. The start precedes the C3-2
requirement of sample start minus the longest candidate lag (1 July 2023 −
119 days = 4 March 2023). The four series are summed and step-expanded to
daily frequency, constant within each week (C3-3).

**Observed publication delay (C3-4).** At retrieval on 20 August 2026 the
most recent week available was 10–16 August 2026, giving an observed delay of
at most four days from week end. The 28-day lag floor exceeds this by a wide
margin: the week supplying a value at date *t* ends no later than *t* − 22 and
was published by approximately *t* − 18.

## File format notes

The Wordstat exports use a UTF-8 byte-order mark, semicolon delimiters,
carriage-return line endings, a space as thousands separator and a comma as
decimal separator. Parsing is handled in `src/align.py`; the files are stored
byte-for-byte as downloaded and are not edited.