-- Display transcript reconstructed from Retell word timestamps.
-- NULL means "no reconstruction available" — the dashboard falls back to the
-- raw `transcript` column, which is always stored untouched.
alter table public.calls
  add column if not exists transcript_display jsonb;
