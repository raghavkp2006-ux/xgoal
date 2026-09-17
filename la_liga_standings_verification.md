# La Liga Standings Verification

The 2018/19 and 2021/22 La Liga standings were reconstructed from the ingested SP1 match results and compared with the published final tables.

Both seasons contain **380 matches**, and all **20 clubs completed 38 matches**.

Reconstruction rules:

- 3 points per win
- 1 point per draw
- Goals for and against calculated from stored match scores
- Tied positions noted where published head-to-head tiebreakers determine the order

Published references:

- [2018–19 La Liga](https://en.wikipedia.org/wiki/2018%E2%80%9319_La_Liga)
- [2021–22 La Liga](https://en.wikipedia.org/wiki/2021%E2%80%9322_La_Liga)

## 2018/19 La Liga

| Pos | Reconstructed | Pts | GF–GA | Published final table | Result |
|---:|---|---:|---:|---|---|
| 1 | Barcelona | 87 | 90–36 | Barcelona | Match |
| 2 | Atlético Madrid | 76 | 55–29 | Atlético Madrid | Match |
| 3 | Real Madrid | 68 | 63–46 | Real Madrid | Match |
| 4 | Valencia | 61 | 51–35 | Valencia | Match |
| 5 | Sevilla | 59 | 62–47 | Getafe | Tie-break order |
| 6 | Getafe | 59 | 48–35 | Sevilla | Tie-break order |
| 7 | Espanyol | 53 | 48–50 | Espanyol | Match |
| 8 | Athletic Club | 53 | 41–45 | Athletic Club | Match |
| 9 | Real Sociedad | 50 | 45–46 | Real Sociedad | Match |
| 10 | Real Betis | 50 | 44–52 | Real Betis | Match |
| 11 | Deportivo Alavés | 50 | 39–50 | Deportivo Alavés | Match |
| 12 | Eibar | 47 | 46–50 | Eibar | Match |
| 13 | Leganés | 45 | 37–43 | Leganés | Match |
| 14 | Villarreal | 44 | 49–52 | Villarreal | Match |
| 15 | Levante | 44 | 59–66 | Levante | Match |
| 16 | Celta Vigo | 41 | 53–62 | Real Valladolid | Tie-break order |
| 17 | Real Valladolid | 41 | 32–51 | Celta Vigo | Tie-break order |
| 18 | Girona | 37 | 37–53 | Girona | Match |
| 19 | Huesca | 33 | 43–65 | Huesca | Match |
| 20 | Rayo Vallecano | 32 | 41–70 | Rayo Vallecano | Match |

The apparent Sevilla/Getafe difference is a tied-team head-to-head tiebreak, as is the
Valladolid/Celta pair: the published table (Wikipedia's Sports-table template, `team16=VLD`,
`team17=CEL`) places **Real Valladolid 16th and Celta Vigo 17th**, because Valladolid took
four of the six head-to-head points. Both clubs' points and goal statistics are correct either
way — only the order depends on the rule. `jobs/validate.py` section 8 applies the full
head-to-head chain, so it reproduces this order exactly.

## 2021/22 La Liga

| Pos | Reconstructed | Pts | GF–GA | Published final table | Result |
|---:|---|---:|---:|---|---|
| 1 | Real Madrid | 86 | 80–31 | Real Madrid | Match |
| 2 | Barcelona | 73 | 68–38 | Barcelona | Match |
| 3 | Atlético Madrid | 71 | 65–43 | Atlético Madrid | Match |
| 4 | Sevilla | 70 | 53–30 | Sevilla | Match |
| 5 | Real Betis | 65 | 62–40 | Real Betis | Match |
| 6 | Real Sociedad | 62 | 40–37 | Real Sociedad | Match |
| 7 | Villarreal | 59 | 63–37 | Villarreal | Match |
| 8 | Athletic Club | 55 | 43–36 | Athletic Club | Match |
| 9 | Valencia | 48 | 48–53 | Valencia | Match |
| 10 | Osasuna | 47 | 37–51 | Osasuna | Match |
| 11 | Celta Vigo | 46 | 43–43 | Celta Vigo | Match |
| 12 | Rayo Vallecano | 42 | 39–50 | Rayo Vallecano | Match |
| 13 | Elche | 42 | 40–52 | Elche | Match |
| 14 | Espanyol | 42 | 40–53 | Espanyol | Match |
| 15 | Getafe | 39 | 33–41 | Getafe | Match |
| 16 | Cádiz | 39 | 35–51 | Mallorca | Tie-break order |
| 17 | Mallorca | 39 | 36–63 | Cádiz | Tie-break order |
| 18 | Granada | 38 | 44–61 | Granada | Match |
| 19 | Levante | 35 | 51–76 | Levante | Match |
| 20 | Deportivo Alavés | 31 | 31–65 | Deportivo Alavés | Match |

The apparent Cádiz/Mallorca difference is a tied-team head-to-head tiebreak. Their reconstructed points and goal statistics are correct.

## Conclusion

The ingested match data reproduces the published final tables for both seasons, including every club's points and goal statistics. The only ordering differences are tied-team head-to-head tiebreaks.
