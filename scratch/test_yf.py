import yfinance as yf
from datetime import datetime

ticker = "^NSEI"
start = "2023-01-01"
end = "2026-05-10"

print(f"Downloading {ticker} from {start} to {end}...")
df = yf.download(ticker, start=start, end=end)
print(f"Download complete. Shape: {df.shape}")
if not df.empty:
    print(f"Last date: {df.index.max()}")
else:
    print("DataFrame is empty!")
