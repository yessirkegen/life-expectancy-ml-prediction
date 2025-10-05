# build_wb_indicator.py
import os, requests, pandas as pd

def fetch_wb_countries():
    """Белый список ISO3 стран (без агрегатов/регионов)."""
    iso3 = set()
    page = 1
    while True:
        r = requests.get("https://api.worldbank.org/v2/country",
                         params={"format":"json","per_page":400,"page":page}, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            break
        for it in data[1]:
            # Берём только настоящие страны (region.value != 'Aggregates')
            if it.get("region", {}).get("value") != "Aggregates":
                iso3.add(it["id"])  # 'id' = ISO3
        if page >= data[0].get("pages", 1):
            break
        page += 1
    return iso3

COUNTRY_WHITELIST = fetch_wb_countries()

def fetch_wb_indicator(indicator: str, value_name: str, years=range(2010, 2020)):
    rows, page = [], 1
    while True:
        url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        params = {"format": "json", "per_page": 20000, "page": page}
        r = requests.get(url, params=params, timeout=60); r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            break
        rows.extend(data[1])
        if page >= data[0].get("pages", 1): break
        page += 1

    df = pd.DataFrame(rows)

    # ISO3
    df["iso3"] = df["countryiso3code"].astype(str).str.strip()

    # Фильтр только на реальные страны по белому списку
    df = df[df["iso3"].isin(COUNTRY_WHITELIST)].copy()

    # Поля
    df["year"] = pd.to_numeric(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df["value"], errors="coerce")

    # Оставляем только нужные годы и непустые значения
    df = df[["iso3","year",value_name]].dropna()
    df = df[df["year"].isin(years)].copy()
    return df.sort_values(["iso3","year"]).reset_index(drop=True)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    # Алкоголь (литры чистого алкоголя на 15+ в год)
    alcohol = fetch_wb_indicator("SH.ALC.PCAP.LI", "alcohol_lpa")
    out = "data/raw/alcohol.csv"
    alcohol.to_csv(out, index=False)
    print("Shape:", alcohol.shape)
    print(alcohol.head())
    print("Saved ->", out)

    # PM2.5 (μg/m³), среднегодовая экспозиция
    pm25 = fetch_wb_indicator("EN.ATM.PM25.MC.M3", "pm25_ugm3")
    pm25.to_csv("data/raw/pm25.csv", index=False)
    print("PM2.5 Shape:", pm25.shape)
    print(pm25.head())
    print("Saved -> data/raw/pm25.csv")

    # GDP per capita (current US$)
    gdp = fetch_wb_indicator("NY.GDP.PCAP.CD", "gdp_pc_usd")
    gdp.to_csv("data/raw/gdp_pc.csv", index=False)
    print("GDP Shape:", gdp.shape)
    print(gdp.head())
    print("Saved -> data/raw/gdp_pc.csv")

    # Life expectancy at birth, total (years)
    le = fetch_wb_indicator("SP.DYN.LE00.IN", "life_expectancy")
    le.to_csv("data/raw/life_expectancy.csv", index=False)
    print("LE Shape:", le.shape)
    print(le.head())
    print("Saved -> data/raw/life_expectancy.csv")
