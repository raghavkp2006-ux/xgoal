"""Division codes for football-data.co.uk multi-league ingestion (M9)."""
from dataclasses import dataclass

@dataclass
class Division:
    code: str
    name: str
    country: str
    tier: int

# All available divisions for the training corpus
DIVISIONS: list[Division] = [
    # Spain
    Division("SP1", "La Liga", "Spain", 1),
    Division("SP2", "Segunda División", "Spain", 2),
    # England
    Division("E0", "Premier League", "England", 1),
    Division("E1", "Championship", "England", 2),
    Division("E2", "League One", "England", 3),
    Division("E3", "League Two", "England", 4),
    # Germany
    Division("D1", "Bundesliga", "Germany", 1),
    Division("D2", "2. Bundesliga", "Germany", 2),
    # Italy
    Division("I1", "Serie A", "Italy", 1),
    Division("I2", "Serie B", "Italy", 2),
    # France
    Division("F1", "Ligue 1", "France", 1),
    Division("F2", "Ligue 2", "France", 2),
    # Netherlands
    Division("N1", "Eredivisie", "Netherlands", 1),
    # Portugal
    Division("P1", "Primeira Liga", "Portugal", 1),
    # Turkey
    Division("T1", "Süper Lig", "Turkey", 1),
    # Belgium
    Division("B1", "Jupiler Pro League", "Belgium", 1),
    # Greece
    Division("G1", "Super League", "Greece", 1),
    # Scotland
    Division("SC0", "Scottish Premiership", "Scotland", 1),
    Division("SC1", "Scottish Championship", "Scotland", 2),
]

SEASON_COUNT = 10  # 2015/16 to 2024/25