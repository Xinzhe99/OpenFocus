CREATE TABLE website_traffic_daily (
  day TEXT PRIMARY KEY NOT NULL,
  page_views INTEGER NOT NULL CHECK (page_views >= 0),
  download_starts INTEGER NOT NULL CHECK (download_starts >= 0)
) WITHOUT ROWID;
