alter table public.calls
  add column if not exists final_outcome text,
  add column if not exists trial_booked boolean default false,
  add column if not exists trial_cancelled boolean default false,
  add column if not exists needs_follow_up boolean default false,
  add column if not exists follow_up_reason text,
  add column if not exists bookings jsonb default '[]'::jsonb;

create index if not exists calls_final_outcome_idx on public.calls(final_outcome);
create index if not exists calls_needs_follow_up_idx on public.calls(needs_follow_up);
