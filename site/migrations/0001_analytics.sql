CREATE TABLE download_counts (
  day TEXT NOT NULL,
  version TEXT NOT NULL,
  kind TEXT NOT NULL,
  channel TEXT NOT NULL,
  source TEXT NOT NULL,
  count INTEGER NOT NULL,
  PRIMARY KEY (day, version, kind, channel, source)
);

CREATE TABLE active_installations (
  day TEXT NOT NULL,
  installation_id TEXT NOT NULL,
  version TEXT NOT NULL,
  platform TEXT NOT NULL,
  PRIMARY KEY (day, installation_id, version, platform)
);
