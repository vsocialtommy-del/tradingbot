-- 017: log into an MT5 account from the dashboard + calendar per account
-- (12 Nov 2026). Run in BOTH Supabase projects (M5 bot and M1 bot).
--
--   bot_commands.payload   the scrambled login sent from the phone; only the
--                          laptop can read it, and it's wiped once used
--   bot_commands.command   now also allows 'login'
--   closed_trades.account  MT5 account number, so the calendar shows the
--                          account the bot is logged into

ALTER TABLE bot_commands ADD COLUMN IF NOT EXISTS payload TEXT;
ALTER TABLE bot_commands DROP CONSTRAINT IF EXISTS bot_commands_command_check;
ALTER TABLE bot_commands ADD CONSTRAINT bot_commands_command_check
  CHECK (command IN ('update', 'restart', 'start', 'stop', 'login'));

ALTER TABLE closed_trades ADD COLUMN IF NOT EXISTS account BIGINT;
CREATE INDEX IF NOT EXISTS idx_closed_trades_account
  ON closed_trades (account, trading_day);
