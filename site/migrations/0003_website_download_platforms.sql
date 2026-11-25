-- NULL preserves dates recorded before downloads were counted by platform.
ALTER TABLE website_traffic_daily ADD COLUMN windows_downloads INTEGER CHECK (windows_downloads >= 0);
ALTER TABLE website_traffic_daily ADD COLUMN macos_downloads INTEGER CHECK (macos_downloads >= 0);
ALTER TABLE website_traffic_daily ADD COLUMN linux_downloads INTEGER CHECK (linux_downloads >= 0);
