"""Fetch S&P 500 ticker list from Wikipedia (current + historical changes)."""
import urllib.request
import re
import os
import json

OUTPUT_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_url(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode('utf-8')


def parse_current_sp500(html):
    """Find tbody → first tr in table, extract ticker (Symbol column)."""
    # The current S&P 500 table has id="constituents"
    # Each row: <tr><td><a>SYMBOL</a></td>...</tr>
    # We use a robust regex
    table_start = html.find('id="constituents"')
    if table_start == -1:
        # Fallback: first wikitable
        table_start = html.find('class="wikitable')
    if table_start == -1:
        return []

    # Find table end
    table_end = html.find('</table>', table_start)
    table_html = html[table_start:table_end]

    # Each row starts with <tr> then <td><a ...>TICKER</a> or <td>TICKER</td>
    tickers = []
    rows = re.split(r'<tr[^>]*>', table_html)
    for row in rows[1:]:  # skip header
        # First <td> typically contains ticker
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if not cells:
            continue
        first_cell = cells[0]
        # Extract text inside <a>...</a> if exists
        a_match = re.search(r'<a[^>]*>([^<]+)</a>', first_cell)
        if a_match:
            ticker = a_match.group(1).strip()
        else:
            ticker = re.sub(r'<[^>]+>', '', first_cell).strip()
        # Clean: tickers are uppercase letters maybe with . (e.g., BRK.B)
        if re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker):
            tickers.append(ticker)
    return tickers


def main():
    print("Fetching S&P 500 list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = fetch_url(url)
    print(f"  Page length: {len(html):,} bytes")

    current = parse_current_sp500(html)
    print(f"  Current S&P 500 tickers found: {len(current)}")
    if current:
        print(f"  Sample: {current[:10]}, ..., {current[-5:]}")

    # Save list
    out_path = os.path.join(OUTPUT_DIR, 'sp500_current.txt')
    with open(out_path, 'w') as f:
        for t in current:
            f.write(t + '\n')
    print(f"  Saved: {out_path}")

    # Also save ETF list
    etfs = [
        # Defensive
        'GLD',   # Gold
        'TLT',   # 20+ year Treasuries
        'SHY',   # 1-3 year Treasuries
        # Sectors (Select Sector SPDR)
        'XLK',   # Technology
        'XLF',   # Financials
        'XLE',   # Energy
        'XLV',   # Healthcare
        'XLI',   # Industrials
        'XLP',   # Consumer Staples
        'XLY',   # Consumer Discretionary
        'XLU',   # Utilities
        'XLC',   # Communication
        'XLB',   # Materials
        'XLRE',  # Real Estate
        # Special
        'ITA',   # Aerospace & Defense
        'SPY',   # S&P 500 benchmark
        'VXX',   # Volatility
    ]
    out_etf = os.path.join(OUTPUT_DIR, 'etf_list.txt')
    with open(out_etf, 'w') as f:
        for e in etfs:
            f.write(e + '\n')
    print(f"  ETF list ({len(etfs)}): {out_etf}")

    # Combined list
    out_all = os.path.join(OUTPUT_DIR, 'all_tickers.txt')
    all_t = current + etfs
    with open(out_all, 'w') as f:
        for t in all_t:
            f.write(t + '\n')
    print(f"  Combined ({len(all_t)}): {out_all}")


if __name__ == '__main__':
    main()
